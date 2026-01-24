import torch
import folder_paths
import comfy.utils
from .lora_utils import parse_wan_lora

class WanLoRALoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "WanVideo/INT8"

    def load_lora(self, model, lora_name, strength):
        if strength == 0:
            return (model,)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            print(f"LoRA not found: {lora_name}")
            return (model,)

        print(f"Loading LoRA: {lora_path} with strength {strength}")
        lora_state_dict = comfy.utils.load_torch_file(lora_path)
        
        # Parse LoRA weights and map to model keys
        lora_weights = parse_wan_lora(lora_state_dict, strength)
        
        # Clone model to avoid mutating the original patcher
        new_model = model.clone()
        
        # Get the underlying torch model
        # In ComfyUI, model.model is the BaseModel
        torch_model = new_model.model
        
        # Map of module name -> module
        modules = dict(torch_model.named_modules())
        
        patched_count = 0
        failed_count = 0
        dim_mismatch_count = 0
        
        for key in lora_weights.weights:
            target_module = None
            target_key = None
            
            # 1. Try exact match
            if key in modules:
                target_module = modules[key]
                target_key = key
            
            # 2. Try with/without 'diffusion_model.' prefix
            if target_module is None:
                if key.startswith("diffusion_model."):
                    alt_key = key[len("diffusion_model."):]
                else:
                    alt_key = "diffusion_model." + key
                
                if alt_key in modules:
                    target_module = modules[alt_key]
                    target_key = alt_key

            # 3. Try replacing .self_attn. with .attn. or vice versa
            if target_module is None:
                alt_key = key
                if ".self_attn." in key:
                    alt_key = key.replace(".self_attn.", ".attn.")
                elif ".attn." in key:
                    alt_key = key.replace(".attn.", ".self_attn.")
                
                if alt_key in modules:
                    target_module = modules[alt_key]
                    target_key = alt_key
                else:
                    # Try prefix variations on the attn-swapped key
                    if alt_key.startswith("diffusion_model."):
                        alt_key2 = alt_key[len("diffusion_model."):]
                    else:
                        alt_key2 = "diffusion_model." + alt_key
                    
                    if alt_key2 in modules:
                        target_module = modules[alt_key2]
                        target_key = alt_key2

            # 4. Try removing .0 from to_out.0
            if target_module is None and ".to_out.0" in key:
                alt_key = key.replace(".to_out.0", ".to_out")
                if alt_key in modules:
                    target_module = modules[alt_key]
                    target_key = alt_key
                elif ("diffusion_model." + alt_key) in modules:
                    target_key = "diffusion_model." + alt_key
                    target_module = modules[target_key]

            # 5. Try mapping to_q -> q, to_k -> k, etc.
            if target_module is None:
                alt_key = key
                replacements = {
                    ".to_q": ".q", ".to_k": ".k", ".to_v": ".v", ".to_out": ".o",
                    ".q": ".to_q", ".k": ".to_k", ".v": ".to_v", ".o": ".to_out"
                }
                for old, new in replacements.items():
                    if old in key:
                        alt_key = key.replace(old, new)
                        break
                
                if alt_key != key:
                    if alt_key in modules:
                        target_module = modules[alt_key]
                        target_key = alt_key
                    elif ("diffusion_model." + alt_key) in modules:
                        target_key = "diffusion_model." + alt_key
                        target_module = modules[target_key]

            if target_module is not None:
                if hasattr(target_module, "lora_patches"):
                    down, up, alpha = lora_weights.weights[key]
                    
                    # Dimension validation
                    # down: (rank, in_features), up: (out_features, rank)
                    # Linear weight: (out_features, in_features)
                    if hasattr(target_module, "weight"):
                        expected_out, expected_in = target_module.weight.shape
                        actual_out, actual_rank = up.shape
                        actual_rank_down, actual_in = down.shape
                        
                        if expected_out != actual_out or expected_in != actual_in:
                            print(f"  [!] Dimension mismatch for {target_key}:")
                            print(f"      Model: {expected_out}x{expected_in}")
                            print(f"      LoRA:  {actual_out}x{actual_in} (rank {actual_rank})")
                            dim_mismatch_count += 1
                            continue

                    # Get current patches from the module (might be already patched)
                    current_patches = getattr(target_module, "lora_patches", [])
                    
                    # Create a new list with the additional patch
                    new_patches = current_patches + [(down, up, alpha)]
                    
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

        print(f"LoRA Application Summary:")
        print(f"  - Successfully patched: {patched_count} layers")
        if dim_mismatch_count > 0:
            print(f"  - Dimension mismatches: {dim_mismatch_count} (skipped)")
        if failed_count > 0:
            print(f"  - Keys not found in model: {failed_count}")
            
            if patched_count == 0:
                print("\n[!] DEBUG: No layers were patched. Showing first 10 LoRA keys and first 10 Model keys:")
                lora_keys = list(lora_weights.weights.keys())
                print(f"    LoRA keys: {lora_keys[:10]}")
                
                model_keys = [k for k in modules.keys() if "blocks" in k and "weight" not in k]
                print(f"    Model keys (sample): {model_keys[:10]}")
                print("    Check if prefixes (like 'diffusion_model.') or naming (like '.self_attn.' vs '.attn.') match.")
                
        return (new_model,)
