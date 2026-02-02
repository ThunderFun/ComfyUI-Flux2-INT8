import os
import torch
import torch.nn.functional as F
import folder_paths
import comfy.utils
import gc
from .lora_utils import parse_wan_lora

# CUDA Synchronization control
_ENABLE_CUDA_SYNC = os.environ.get("INT8_ENABLE_CUDA_SYNC", "0") == "1"
_CLEAR_CACHE_STRATEGY = os.environ.get("INT8_CLEAR_CACHE", "auto")

# LoRA weight cache size limit (number of cached LoRA patches)
_MAX_LORA_CACHE_SIZE = int(os.environ.get("INT8_LORA_CACHE_SIZE", "32"))


def _should_clear_cache():
    """Check if we should clear CUDA cache based on memory pressure."""
    if _CLEAR_CACHE_STRATEGY == "always":
        return True
    if _CLEAR_CACHE_STRATEGY == "never":
        return False
    if not torch.cuda.is_available():
        return False
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    if reserved > 0:
        utilization = allocated / reserved
                # This prevents unnecessary clearing when memory is actually being used
        return utilization < 0.5
    return False

try:
    from .int8_quant import Int8TensorwiseOps
except ImportError:
    Int8TensorwiseOps = None

class LoRAWeightCache:
    """LRU cache for LoRA weights with automatic cleanup."""
    
    def __init__(self, max_size=_MAX_LORA_CACHE_SIZE):
        self.max_size = max_size
        self._cache = {}
        self._access_order = []
    
    def get(self, key):
        if key in self._cache:
            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def set(self, key, value):
        if key in self._cache:
            # Update existing
            self._cache[key] = value
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
        else:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size:
                self._evict_oldest()
            self._cache[key] = value
            self._access_order.append(key)
    
    def _evict_oldest(self):
        if self._access_order:
            oldest = self._access_order.pop(0)
            if oldest in self._cache:
                del self._cache[oldest]
    
    def clear(self):
        """Clear all cached weights and free memory."""
        self._cache.clear()
        self._access_order.clear()
    
    def __len__(self):
        return len(self._cache)


def make_patched_forward(mod, orig_fwd):
    # Attach cache to module for external access and cleanup
    if not hasattr(mod, '_lora_weight_cache'):
        mod._lora_weight_cache = LoRAWeightCache(_MAX_LORA_CACHE_SIZE)
    
    def patched_forward(x):
        out = orig_fwd(x)
        patches = getattr(mod, "lora_patches", [])
        if patches:
            # Clear cache if patches have changed (different LoRA loaded)
            def patch_hash(p):
                """Create a hash based on patch tensor shapes and dtypes, not memory address."""
                d, u = p[0], p[1]
                return hash((d.shape, d.dtype, u.shape, u.dtype, id(p)))  # id(p) as fallback entropy
            
            current_patch_ids = tuple(patch_hash(p) for p in patches)
            last_patch_ids = getattr(mod, '_last_patch_ids', None)
            if last_patch_ids is not None and last_patch_ids != current_patch_ids:
                # Patches changed - clear the cache to prevent memory leak
                mod._lora_weight_cache.clear()
            mod._last_patch_ids = current_patch_ids
            
            for patch_data in patches:
                # Unpack patch data (supports both old and new format)
                if len(patch_data) == 3:
                    # Old format: (down, up, alpha)
                    d, u, a = patch_data
                    d_scale, u_scale = None, None
                    offset, size = 0, 0
                elif len(patch_data) == 5:
                    # New format: (down, up, alpha, down_scale, up_scale)
                    d, u, a, d_scale, u_scale = patch_data
                    offset, size = 0, 0
                else:
                    # Extended format: (down, up, alpha, down_scale, up_scale, offset, size)
                    d, u, a, d_scale, u_scale, offset, size = patch_data
                
                # Check if this is INT8 LoRA
                is_int8 = d.dtype == torch.int8 and u.dtype == torch.int8
                
                if is_int8 and d_scale is not None and u_scale is not None:
                    # INT8 LoRA path using torch._int_mm
                    from .int8_quant import chunked_int8_lora_forward, CHUNK_THRESHOLD_ELEMENTS
                    
                    # Flatten x to 2D for matmul
                    x_shape = x.shape
                    x_2d = x.reshape(-1, x_shape[-1])
                    
                                        # Use hash of full tensor bytes for small tensors, sampled hash + checksum for large
                    def tensor_content_hash(t):
                        """Create a strong hash based on tensor content."""
                        if t.numel() == 0:
                            return 0
                        
                        # For small tensors, hash the full content
                        if t.numel() <= 1024:
                            return hash((t.tobytes(), t.shape, t.dtype, t.device.index if t.device.type == 'cuda' else -1))
                        
                        # For large tensors, use multiple samples + statistical summary
                        flat = t.flatten()
                        n = flat.numel()
                        
                        # Sample at multiple points with denser sampling at boundaries
                        sample_indices = [
                            0, 1, 2,  # Start
                            n // 8, n // 4, n // 2, 3 * n // 4, 7 * n // 8,  # Middle regions
                            n - 3, n - 2, n - 1  # End
                        ]
                        sample_indices = list(dict.fromkeys(i for i in sample_indices if 0 <= i < n))  # Unique valid indices
                        samples = flat[sample_indices].tolist()
                        
                        # Add statistical summary to detect distribution changes
                        mean_val = flat.mean().item()
                        std_val = flat.std().item()
                        
                        return hash((tuple(samples), mean_val, std_val, t.shape, t.dtype, t.device.index if t.device.type == 'cuda' else -1))
                    
                    cache_key = (d.shape, d.dtype, u.shape, u.dtype, tensor_content_hash(d), tensor_content_hash(u))
                    
                    cached = mod._lora_weight_cache.get(cache_key)
                    if cached is None:
                        # First time - move to GPU and cache
                        cached = {
                            'd': d.to(device=out.device, non_blocking=True),
                            'u': u.to(device=out.device, non_blocking=True),
                            'd_scale': d_scale.to(device=out.device, non_blocking=True) if isinstance(d_scale, torch.Tensor) else d_scale,
                            'u_scale': u_scale.to(device=out.device, non_blocking=True) if isinstance(u_scale, torch.Tensor) else u_scale,
                        }
                        mod._lora_weight_cache.set(cache_key, cached)
                    
                    # Use cached weights
                    chunked_int8_lora_forward(
                        x_2d, cached['d'], cached['u'], 
                        cached['d_scale'], cached['u_scale'], 
                        a, out,
                        offset=offset, size=size
                    )
                    
                    # Clear cache based on memory pressure strategy
                    if out.numel() > CHUNK_THRESHOLD_ELEMENTS and _should_clear_cache():
                        mod._lora_weight_cache.clear()
                        torch.cuda.empty_cache()
                    
                    del x_2d
                else:
                    # Float LoRA path
                    from .int8_quant import chunked_lora_forward
                    
                    # Flatten x to 2D for matmul (consistent with INT8 path)
                    x_shape = x.shape
                    x_2d = x.reshape(-1, x_shape[-1])
                    
                    # Ensure x matches output dtype for LoRA math
                    if x_2d.dtype != out.dtype:
                        x_2d = x_2d.to(dtype=out.dtype)
                    
                    d_t = d.to(device=out.device, dtype=out.dtype)
                    u_t = u.to(device=out.device, dtype=out.dtype)
                    
                    # Use memory-efficient chunked forward
                    chunked_lora_forward(x_2d, d_t, u_t, a, out, offset=offset, size=size)
                    
                    del d_t, u_t, x_2d
        return out
    return patched_forward

class WanLoRALoaderWithCLIP:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "offload_to_cpu": (["enable", "disable"], {"default": "disable"}),
            },
            "optional": {
                "clip": ("CLIP",),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "WanVideo/INT8"
    
    def load_lora(self, model, lora_name, strength_model, strength_clip, clip=None, offload_to_cpu="disable", debug=False):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            print(f"LoRA not found: {lora_name}")
            return (model, clip)

        print(f"Loading LoRA: {lora_name}")
        
        # Determine if we should offload float LoRAs to CPU
        offload_enabled = (offload_to_cpu == "enable")
        
        lora_state_dict = comfy.utils.load_torch_file(lora_path)

        # Parse LoRA weights with strength 1.0, we'll apply strengths during patching
        lora_weights = parse_wan_lora(lora_state_dict, 1.0, debug=debug)
        
        # Clear state dict immediately to save memory
        del lora_state_dict
        gc.collect()
        
        # Patch Model
        new_model = model.clone()
        if strength_model != 0:
            self.patch_patcher(new_model, lora_weights, strength_model, offload_enabled, debug, is_clip=False)
            
        # Patch CLIP
        if clip is not None:
            new_clip = clip.clone()
            if strength_clip != 0:
                # Check if LoRA contains any CLIP keys before attempting to patch
                has_clip_keys = any("qwen3_4b" in k or "text_encoder" in k or ".layers." in k for k in lora_weights.weights.keys())
                
                if has_clip_keys:
                    # CLIP in ComfyUI has a patcher attribute
                    self.patch_patcher(new_clip.patcher, lora_weights, strength_clip, offload_enabled, debug, is_clip=True)
                else:
                    if debug:
                        print("Skipping CLIP patching: No CLIP-specific keys found in LoRA")
        else:
            new_clip = None
            
        # Clear modules dict and collect garbage
        gc.collect()
                
        return (new_model, new_clip)

    def patch_patcher(self, patcher, lora_weights, strength, offload_enabled, debug, is_clip=False):
        # Get the underlying torch model
        torch_model = patcher.model
        
        # Map of module name -> module
        modules = dict(torch_model.named_modules())
        
        if debug:
            type_name = "CLIP" if is_clip else "Model"
            print(f"\n[DEBUG] {type_name} Available modules ({len(modules)}):")
            linear_modules = [k for k, v in modules.items() if isinstance(v, torch.nn.Linear)]
            print(f"[DEBUG] {type_name} Linear modules: {len(linear_modules)}")
            for i, key in enumerate(sorted(linear_modules)[:50]):
                print(f"  {i+1}. {key}")
            if len(linear_modules) > 50:
                print(f"  ... and {len(linear_modules) - 50} more")

        patched_count = 0
        failed_count = 0
        dim_mismatch_count = 0
        failed_keys = []
        
        device = getattr(patcher, "load_device", torch.device("cpu"))

        for key in lora_weights.weights:
            # Skip diffusion model keys when patching CLIP
            if is_clip and key.startswith("diffusion_model."):
                continue
                
            target_module = None
            target_key = None
            
            
            # Generate candidate keys
            candidates = [key]
            
            if not is_clip:
                # UNet variations
                if key.startswith("diffusion_model."):
                    candidates.append(key[len("diffusion_model."):])
                else:
                    candidates.append("diffusion_model." + key)
            else:
                # CLIP variations
                # Try stripping common prefixes
                for prefix in ["text_encoders.", "lora_te.", "lora_te1.", "lora_te2.", "te_model."]:
                    if key.startswith(prefix):
                        candidates.append(key[len(prefix):])

            # Common variations and replacements
            # We use a list of potential transformations to avoid circular replacements
            transformations = [
                (".self_attn.", ".attn."), (".attn.", ".self_attn."),
                (".to_out.0", ".to_out"), (".to_out", ".to_out.0"),
                (".to_q", ".q"), (".to_k", ".k"), (".to_v", ".v"), (".to_out", ".o"),
                (".q", ".to_q"), (".k", ".to_k"), (".v", ".to_v"), (".o", ".to_out"),
                (".to_q", ".qkv"), (".to_k", ".qkv"), (".to_v", ".qkv"),
                (".to_out.0", ".out"), (".to_out", ".out"),
                (".q_proj", ".qkv"), (".k_proj", ".qkv"), (".v_proj", ".qkv"),
            ]
            
            current_candidates = set(candidates)
            # Apply transformations iteratively to handle multiple changes (e.g. layers->blocks AND attention->self_attn)
            for _ in range(3):
                new_cands = set()
                for c in current_candidates:
                    for old, new in transformations:
                        if old in c:
                            new_cands.add(c.replace(old, new))
                if not new_cands:
                    break
                if new_cands.issubset(current_candidates):
                    break
                current_candidates.update(new_cands)
            # Add diffusion_model prefix back to all candidates if it was there originally
            if key.startswith("diffusion_model."):
                for c in list(current_candidates):
                    if not c.startswith("diffusion_model."):
                        current_candidates.add("diffusion_model." + c)
            
            candidates = list(current_candidates)
            
            # Remove duplicates and try matching
            patch_offset = 0
            patch_size = 0
            
            seen = set()
            for cand in candidates:
                if cand in seen: continue
                seen.add(cand)
                if cand in modules:
                    target_module = modules[cand]
                    target_key = cand
                    
                    # Handle fused QKV mapping
                    if ".qkv" in cand:
                        if ".to_q" in key or ".q_proj" in key or "_attn_q" in key:
                            patch_size = target_module.weight.shape[0] // 3
                            patch_offset = 0
                        elif ".to_k" in key or ".k_proj" in key or "_attn_k" in key:
                            patch_size = target_module.weight.shape[0] // 3
                            patch_offset = patch_size
                        elif ".to_v" in key or ".v_proj" in key or "_attn_v" in key:
                            patch_size = target_module.weight.shape[0] // 3
                            patch_offset = patch_size * 2
                    
                    break
            
            # Fallback for CLIP: try underscore to dot conversion
            if is_clip and target_module is None:
                for cand in candidates:
                    if "_" in cand:
                        dot_cand = cand.replace("_", ".")
                        if dot_cand in modules:
                            target_module = modules[dot_cand]
                            target_key = dot_cand
                            break

            if target_module is not None:
                # If module doesn't have lora_patches, try to add it and patch forward
                if not hasattr(target_module, "lora_patches"):
                    if isinstance(target_module, torch.nn.Linear):
                        target_module.lora_patches = []
                        original_forward = target_module.forward
                        target_module.forward = make_patched_forward(target_module, original_forward)

                if hasattr(target_module, "lora_patches"):
                    is_int8 = lora_weights.is_int8.get(key, False)
                    if is_int8:
                        down, up, down_scale, up_scale, alpha = lora_weights.weights[key]
                    else:
                        down, up, alpha = lora_weights.weights[key]
                        down_scale = None
                        up_scale = None
                    
                    alpha = alpha * strength
                    
                    if hasattr(target_module, "weight"):
                        expected_out, expected_in = target_module.weight.shape
                        actual_out, actual_rank = up.shape
                        actual_rank_down, actual_in = down.shape
                        
                        # Adjust expected_out if we are patching a slice
                        validation_out = patch_size if patch_size > 0 else expected_out
                        
                        if validation_out != actual_out or expected_in != actual_in:
                            if debug:
                                print(f"  [!] Dimension mismatch for {target_key}: Model {validation_out}x{expected_in} (Total: {expected_out}), LoRA {actual_out}x{actual_in}")
                            dim_mismatch_count += 1
                            continue
                            
                    current_patches = getattr(target_module, "lora_patches", [])
                    if is_int8 or not offload_enabled:
                        down = down.to(device=device, non_blocking=True)
                        up = up.to(device=device, non_blocking=True)
                        if isinstance(down_scale, torch.Tensor):
                            down_scale = down_scale.to(device=device, non_blocking=True)
                        if isinstance(up_scale, torch.Tensor):
                            up_scale = up_scale.to(device=device, non_blocking=True)
                    
                    if patch_size > 0 and hasattr(target_module, "weight"):
                        expected_out = target_module.weight.shape[0]
                        if patch_offset + patch_size > expected_out:
                            print(f"  [!] Bounds check failed for {target_key}: offset={patch_offset}, size={patch_size}, output_dim={expected_out}")
                            # Clamp to valid range
                            patch_size = max(0, expected_out - patch_offset)
                            if patch_size <= 0:
                                print(f"  [!] Skipping patch for {target_key}: size would be <=0 after clamping")
                                continue
                            print(f"  [!] Clamped patch_size to {patch_size} for {target_key}")
                    
                    if patch_size <= 0 and patch_offset > 0:
                        print(f"  [!] Skipping patch for {target_key}: invalid patch size {patch_size}")
                        continue
                    
                    # Create a new list with the additional patch
                    # Store as (down, up, alpha, down_scale, up_scale, offset, size)
                    patch_tuple = (down, up, alpha, down_scale, up_scale, patch_offset, patch_size)
                    new_patches = current_patches + [patch_tuple]
                    target_module.lora_patches = new_patches
                    try:
                        patcher.set_model_patch_replace(new_patches, target_key, "lora_patches")
                    except Exception:
                        pass
                    patched_count += 1
            else:
                failed_count += 1
                failed_keys.append(key)

        type_name = "CLIP" if is_clip else "Model"
        print(f"{type_name} LoRA Application Summary: Patched {patched_count} layers, {failed_count} keys not found")
        if debug and failed_count > 0:
            print(f"[DEBUG] {type_name} First 5 failed keys: {failed_keys[:5]}")
