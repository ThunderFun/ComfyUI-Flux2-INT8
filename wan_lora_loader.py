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

class WanLoRALoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "offload_to_cpu": (["enable", "disable"], {"default": "disable"}),
            },
            "optional": {
                "debug": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "WanVideo/INT8"
    
    def load_lora(self, model, lora_name, strength, offload_to_cpu="disable", debug=False):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if strength == 0:
            return (model,)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            print(f"LoRA not found: {lora_name}")
            return (model,)

        print(f"Loading LoRA: {lora_name}")
        
        # Determine if we should offload float LoRAs to CPU
        offload_enabled = (offload_to_cpu == "enable")
        
        lora_state_dict = comfy.utils.load_torch_file(lora_path)

        # Parse LoRA weights and map to model keys
        lora_weights = parse_wan_lora(lora_state_dict, strength, debug=debug)
        
        # Clear state dict immediately to save memory
        del lora_state_dict
        gc.collect()
        
        # Clone model to avoid mutating the original patcher
        new_model = model.clone()
        
        # Get the underlying torch model
        # In ComfyUI, model.model is the BaseModel
        torch_model = new_model.model
        
        # Map of module name -> module
        modules = dict(torch_model.named_modules())
        
        if debug:
            print(f"\n[DEBUG] Available model modules ({len(modules)}):")
            linear_modules = [k for k, v in modules.items() if isinstance(v, torch.nn.Linear)]
            print(f"[DEBUG] Linear modules: {len(linear_modules)}")
            for i, key in enumerate(sorted(linear_modules)[:20]):  # Show first 20 Linear modules
                mod = modules[key]
                print(f"  {i+1}. {key} ({mod.weight.shape[0]}x{mod.weight.shape[1]})")
            if len(linear_modules) > 20:
                print(f"  ... and {len(linear_modules) - 20} more Linear modules")
        
        patched_count = 0
        failed_count = 0
        
        failed_keys = []  # Track which keys failed
        dim_mismatch_count = 0
        
        for key in lora_weights.weights:
            target_module = None
            target_key = None
            
            if debug:
                print(f"\n[DEBUG] Processing LoRA key: {key}")
            
            # Generate candidate keys
            candidates = [key]
            
            # 1. Prefix variations
            if key.startswith("diffusion_model."):
                candidates.append(key[len("diffusion_model."):])
            else:
                candidates.append("diffusion_model." + key)
            
            # 2. Attn variations
            new_candidates = []
            for c in candidates:
                if ".self_attn." in c:
                    new_candidates.append(c.replace(".self_attn.", ".attn."))
                elif ".attn." in c:
                    new_candidates.append(c.replace(".attn.", ".self_attn."))
            candidates.extend(new_candidates)
            
            # 3. to_out variations
            new_candidates = []
            for c in candidates:
                if ".to_out.0" in c:
                    new_candidates.append(c.replace(".to_out.0", ".to_out"))
                elif ".to_out" in c and ".to_out.0" not in c:
                    new_candidates.append(c.replace(".to_out", ".to_out.0"))
            candidates.extend(new_candidates)
            
            # 4. q/k/v/o variations
            new_candidates = []
            replacements = [
                # Single projection mappings
                (".to_q", ".q"), (".to_k", ".k"), (".to_v", ".v"), (".to_out", ".o"),
                (".q", ".to_q"), (".k", ".to_k"), (".v", ".to_v"), (".o", ".to_out"),
                # Fused QKV mappings
                (".to_q", ".qkv"), (".to_k", ".qkv"), (".to_v", ".qkv"),
                (".q_proj", ".qkv"), (".k_proj", ".qkv"), (".v_proj", ".qkv"),
                # Output mappings
                (".to_out.0", ".out"), (".to_out", ".out"),
            ]
            for c in candidates:
                for old, new in replacements:
                    if old in c:
                        new_candidates.append(c.replace(old, new))
            candidates.extend(new_candidates)
            
            if debug:
                print(f"[DEBUG] Generated {len(candidates)} candidates")
            
            # 5. Remove duplicates and try matching
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
                    
                    if debug:
                        print(f"[DEBUG] ✓ Matched to model key: {target_key} (offset={patch_offset}, size={patch_size})")
                    break

            if target_module is not None:
                # If module doesn't have lora_patches, try to add it and patch forward
                if not hasattr(target_module, "lora_patches"):
                    if isinstance(target_module, torch.nn.Linear):
                        target_module.lora_patches = []
                        
                        # Patch forward method to support lora_patches
                        original_forward = target_module.forward
                        target_module.forward = make_patched_forward(target_module, original_forward)

                if hasattr(target_module, "lora_patches"):
                    # Check if this is INT8 LoRA
                    is_int8 = lora_weights.is_int8.get(key, False)
                    
                    if is_int8:
                        # INT8 LoRA: (down, up, down_scale, up_scale, alpha)
                        down, up, down_scale, up_scale, alpha = lora_weights.weights[key]
                    else:
                        # Float LoRA: (down, up, alpha)
                        down, up, alpha = lora_weights.weights[key]
                        down_scale = None
                        up_scale = None
                    
                    # Dimension validation
                    # down: (rank, in_features), up: (out_features, rank)
                    # Linear weight: (out_features, in_features)
                    if hasattr(target_module, "weight"):
                        expected_out, expected_in = target_module.weight.shape
                        actual_out, actual_rank = up.shape
                        actual_rank_down, actual_in = down.shape
                        
                        # Adjust expected_out if we are patching a slice
                        validation_out = patch_size if patch_size > 0 else expected_out
                        
                        if validation_out != actual_out or expected_in != actual_in:
                            print(f"  [!] Dimension mismatch for {target_key}:")
                            print(f"      Model: {validation_out}x{expected_in} (Total: {expected_out})")
                            print(f"      LoRA:  {actual_out}x{actual_in} (rank {actual_rank})")
                            dim_mismatch_count += 1
                            continue
    
                    # Get current patches from the module (might be already patched)
                    current_patches = getattr(target_module, "lora_patches", [])
                    
                    # Cache weights on device to avoid redundant transfers on every forward
                    # We use the model's load_device if available
                    device = getattr(model, "load_device", torch.device("cpu"))
                    
                    if is_int8 or not offload_enabled:
                        # Move to device during loading if it's INT8 or offloading is disabled
                        down = down.to(device=device, non_blocking=True)
                        up = up.to(device=device, non_blocking=True)
                        if isinstance(down_scale, torch.Tensor):
                            down_scale = down_scale.to(device=device, non_blocking=True)
                        if isinstance(up_scale, torch.Tensor):
                            up_scale = up_scale.to(device=device, non_blocking=True)
                    # If offload_enabled is True and it's a float LoRA, we keep it on CPU to save VRAM
                    # and only move it to device during the forward pass.

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
                    
                    # DIRECTLY set the attribute on the module.
                    # ComfyUI's set_model_patch_replace is for attention processors,
                    # not for arbitrary attributes on INT8 modules.
                    target_module.lora_patches = new_patches
                    
                    # Also register it in the patcher so it's tracked (optional but good for compatibility)
                    try:
                        new_model.set_model_patch_replace(new_patches, target_key, "lora_patches")
                    except Exception:
                        pass
                        
                    patched_count += 1
                else:
                    # Module found but doesn't support lora_patches (e.g. not an INT8 Linear)
                    pass
            else:
                failed_count += 1
                failed_keys.append(key)
                if debug:
                    print(f"[DEBUG] ✗ Failed to match key - tried {len(seen)} candidates")

        print(f"LoRA Application Summary:")
        print(f"  - Successfully patched: {patched_count} layers")
        if dim_mismatch_count > 0:
            print(f"  - Dimension mismatches: {dim_mismatch_count} (skipped)")
        if failed_count > 0:
            print(f"  - Keys not found in model: {failed_count}")
            if debug:
                print(f"\n[DEBUG] Failed keys:")
                for fk in failed_keys:
                    print(f"  - {fk}")
        
        # Clear modules dict and collect garbage
        del modules
        gc.collect()
                
        return (new_model,)
