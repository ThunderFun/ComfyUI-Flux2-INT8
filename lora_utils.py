import torch
from torch import Tensor
import torch.nn.functional as F
import re

# Standard LoRA key patterns for Wan models
WAN_LORA_KEY_MAP = {
    # Transformer blocks (Kohya style)
    r"lora_transformer_blocks_(\d+)_attn_to_q": "diffusion_model.blocks.{i}.self_attn.to_q",
    r"lora_transformer_blocks_(\d+)_attn_to_k": "diffusion_model.blocks.{i}.self_attn.to_k",
    r"lora_transformer_blocks_(\d+)_attn_to_v": "diffusion_model.blocks.{i}.self_attn.to_v",
    r"lora_transformer_blocks_(\d+)_attn_to_out": "diffusion_model.blocks.{i}.self_attn.to_out.0",
    
    # Cross attention
    r"lora_transformer_blocks_(\d+)_cross_attn_to_q": "diffusion_model.blocks.{i}.cross_attn.to_q",
    r"lora_transformer_blocks_(\d+)_cross_attn_to_k": "diffusion_model.blocks.{i}.cross_attn.to_k",
    r"lora_transformer_blocks_(\d+)_cross_attn_to_v": "diffusion_model.blocks.{i}.cross_attn.to_v",
    r"lora_transformer_blocks_(\d+)_cross_attn_to_out": "diffusion_model.blocks.{i}.cross_attn.to_out.0",
    
    # FFN layers
    r"lora_transformer_blocks_(\d+)_ffn_fc1": "diffusion_model.blocks.{i}.ffn.fc1",
    r"lora_transformer_blocks_(\d+)_ffn_fc2": "diffusion_model.blocks.{i}.ffn.fc2",

    # XLabs style mappings
    r"transformer\.blocks\.(\d+)\.attn\.to_q": "diffusion_model.blocks.{i}.self_attn.to_q",
    r"transformer\.blocks\.(\d+)\.attn\.to_k": "diffusion_model.blocks.{i}.self_attn.to_k",
    r"transformer\.blocks\.(\d+)\.attn\.to_v": "diffusion_model.blocks.{i}.self_attn.to_v",
    r"transformer\.blocks\.(\d+)\.attn\.to_out\.0": "diffusion_model.blocks.{i}.self_attn.to_out.0",
    r"transformer\.blocks\.(\d+)\.ffn\.fc1": "diffusion_model.blocks.{i}.ffn.fc1",
    r"transformer\.blocks\.(\d+)\.ffn\.fc2": "diffusion_model.blocks.{i}.ffn.fc2",

    # Simple blocks style (some trainers)
    r"blocks\.(\d+)\.attn\.q_proj": "diffusion_model.blocks.{i}.self_attn.to_q",
    r"blocks\.(\d+)\.attn\.k_proj": "diffusion_model.blocks.{i}.self_attn.to_k",
    r"blocks\.(\d+)\.attn\.v_proj": "diffusion_model.blocks.{i}.self_attn.to_v",
    r"blocks\.(\d+)\.attn\.out_proj": "diffusion_model.blocks.{i}.self_attn.to_out.0",

    # lora_unet_blocks style patterns
    r"lora_unet_blocks_(\d+)_self_attn_q": "diffusion_model.blocks.{i}.self_attn.q",
    r"lora_unet_blocks_(\d+)_self_attn_k": "diffusion_model.blocks.{i}.self_attn.k",
    r"lora_unet_blocks_(\d+)_self_attn_v": "diffusion_model.blocks.{i}.self_attn.v",
    r"lora_unet_blocks_(\d+)_self_attn_o": "diffusion_model.blocks.{i}.self_attn.o",
    
    r"lora_unet_blocks_(\d+)_cross_attn_q": "diffusion_model.blocks.{i}.cross_attn.q",
    r"lora_unet_blocks_(\d+)_cross_attn_k": "diffusion_model.blocks.{i}.cross_attn.k",
    r"lora_unet_blocks_(\d+)_cross_attn_v": "diffusion_model.blocks.{i}.cross_attn.v",
    r"lora_unet_blocks_(\d+)_cross_attn_o": "diffusion_model.blocks.{i}.cross_attn.o",
    
    r"lora_unet_blocks_(\d+)_cross_attn_k_img": "diffusion_model.blocks.{i}.cross_attn.k_img",
    r"lora_unet_blocks_(\d+)_cross_attn_v_img": "diffusion_model.blocks.{i}.cross_attn.v_img",
    r"diffusion_model\.blocks\.(\d+)\.cross_attn\.k_img": "diffusion_model.blocks.{i}.cross_attn.k_img",
    r"diffusion_model\.blocks\.(\d+)\.cross_attn\.v_img": "diffusion_model.blocks.{i}.cross_attn.v_img",
    
    r"lora_unet_blocks_(\d+)_ffn_0": "diffusion_model.blocks.{i}.ffn.0",
    r"lora_unet_blocks_(\d+)_ffn_2": "diffusion_model.blocks.{i}.ffn.2",
    
    r"diffusion_model\.img_emb\.proj\.(\d+)": "diffusion_model.img_emb.proj.{i}",

    # Wan 2.1 / 2.2 PEFT patterns
    r"diffusion_model\.layers\.(\d+)\.attention\.to_q": "diffusion_model.layers.{i}.attention.to_q",
    r"diffusion_model\.layers\.(\d+)\.attention\.to_k": "diffusion_model.layers.{i}.attention.to_k",
    r"diffusion_model\.layers\.(\d+)\.attention\.to_v": "diffusion_model.layers.{i}.attention.to_v",
    r"diffusion_model\.layers\.(\d+)\.attention\.to_out\.0": "diffusion_model.layers.{i}.attention.to_out.0",
    r"diffusion_model\.layers\.(\d+)\.feed_forward\.w1": "diffusion_model.layers.{i}.feed_forward.w1",
    r"diffusion_model\.layers\.(\d+)\.feed_forward\.w2": "diffusion_model.layers.{i}.feed_forward.w2",
    r"diffusion_model\.layers\.(\d+)\.feed_forward\.w3": "diffusion_model.layers.{i}.feed_forward.w3",
    r"diffusion_model\.layers\.(\d+)\.adaLN_modulation\.0": "diffusion_model.layers.{i}.adaLN_modulation.0",

    # Flux patterns (Kohya/X-Labs style)
    r"lora_unet_double_blocks_(\d+)_img_attn_qkv": "diffusion_model.double_blocks.{i}.img_attn.qkv",
    r"lora_unet_double_blocks_(\d+)_img_attn_proj": "diffusion_model.double_blocks.{i}.img_attn.proj",
    r"lora_unet_double_blocks_(\d+)_txt_attn_qkv": "diffusion_model.double_blocks.{i}.txt_attn.qkv",
    r"lora_unet_double_blocks_(\d+)_txt_attn_proj": "diffusion_model.double_blocks.{i}.txt_attn.proj",
    r"lora_unet_double_blocks_(\d+)_img_mlp_0": "diffusion_model.double_blocks.{i}.img_mlp.0",
    r"lora_unet_double_blocks_(\d+)_img_mlp_2": "diffusion_model.double_blocks.{i}.img_mlp.2",
    r"lora_unet_double_blocks_(\d+)_txt_mlp_0": "diffusion_model.double_blocks.{i}.txt_mlp.0",
    r"lora_unet_double_blocks_(\d+)_txt_mlp_2": "diffusion_model.double_blocks.{i}.txt_mlp.2",
    
    r"lora_unet_single_blocks_(\d+)_linear1": "diffusion_model.single_blocks.{i}.linear1",
    r"lora_unet_single_blocks_(\d+)_linear2": "diffusion_model.single_blocks.{i}.linear2",
}

class LoRAWeights:
    def __init__(self):
        self.weights = {}  # key -> (lora_down, lora_up, alpha) or (lora_down, lora_up, down_scale, up_scale, alpha) for INT8
        self.device = torch.device("cpu")
        self.is_int8 = {}  # Track which weights are INT8
    
    def add(self, key: str, down: Tensor, up: Tensor, alpha: float = 1.0, down_scale: Tensor | float = None, up_scale: Tensor | float = None):
        """
        Add LoRA weights. Supports both float and INT8 quantized weights.
        
        Args:
            key: Layer key
            down: Down projection weight (float or INT8)
            up: Up projection weight (float or INT8)
            alpha: LoRA alpha scaling factor
            down_scale: Scale for INT8 down weight (required if down is INT8)
            up_scale: Scale for INT8 up weight (required if up is INT8)
        """
        # Check if this is INT8 LoRA
        is_int8 = down.dtype == torch.int8 and up.dtype == torch.int8
        
        if is_int8:
            if down_scale is None or up_scale is None:
                raise ValueError(f"INT8 LoRA weights require scales. Got down_scale={down_scale}, up_scale={up_scale}")
            
            # Store INT8 weights with their scales
            self.weights[key] = (
                down.to(device=self.device),
                up.to(device=self.device),
                down_scale if isinstance(down_scale, float) else down_scale.to(device=self.device),
                up_scale if isinstance(up_scale, float) else up_scale.to(device=self.device),
                alpha
            )
            self.is_int8[key] = True
        else:
            # Store LoRA tensors in their original dtype (cast at application time)
            # This preserves dtype information and avoids forcing bf16/fp32 -> fp16
            self.weights[key] = (
                down.to(device=self.device),   # keep dtype
                up.to(device=self.device),     # keep dtype
                alpha,
            )
            self.is_int8[key] = False
    
    def get_for_layer(self, key: str, device: torch.device):
        if key not in self.weights:
            return None
        
        w = self.weights[key]
        if self.is_int8[key]:
            # (down, up, down_scale, up_scale, alpha) - see add() method
            return (
                w[0].to(device),  # down
                w[1].to(device),  # up
                w[2].to(device) if isinstance(w[2], torch.Tensor) else w[2],  # down_scale
                w[3].to(device) if isinstance(w[3], torch.Tensor) else w[3],  # up_scale
                w[4]  # alpha
            )
        else:
            # (down, up, alpha)
            return (w[0].to(device), w[1].to(device), w[2])

def detect_lora_format(state_dict):
    keys = list(state_dict.keys())
    
    if any("lora_down" in k for k in keys):
        return "kohya"
    elif any("lora_A" in k for k in keys):
        return "peft"
    elif any(".down.weight" in k for k in keys) or any(".down_proj.weight" in k for k in keys):
        return "standard"
    
    return "unknown"

def parse_wan_lora(state_dict, strength=1.0, debug=False):
    lora_format = detect_lora_format(state_dict)
    print(f"Detected LoRA format: {lora_format}")
    
    if debug:
        print(f"[DEBUG] Original LoRA state_dict keys ({len(state_dict)}):")
        for i, key in enumerate(sorted(state_dict.keys())[:20]):  # Show first 20
            print(f"  {i+1}. {key}")
        if len(state_dict) > 20:
            print(f"  ... and {len(state_dict) - 20} more")
    
    parsed_weights = LoRAWeights()
    
    # Group keys by their base name
    groups = {}
    
    if lora_format == "kohya":
        # Kohya format: lora_transformer_blocks_0_attn_to_q.lora_down.weight
        keys = list(state_dict.keys())
        for key in keys:
            if ".lora_down.weight" in key:
                base_key = key.replace(".lora_down.weight", "")
                down_weight = state_dict.pop(key, None)
                up_weight = state_dict.pop(base_key + ".lora_up.weight", None)
                
                # Skip if either weight is missing
                if down_weight is None or up_weight is None:
                    continue
                    
                alpha = state_dict.pop(base_key + ".alpha", None)
                
                # Check for INT8 scales (pattern: .scale_weight)
                down_scale = state_dict.pop(base_key + ".lora_down.scale_weight", None)
                up_scale = state_dict.pop(base_key + ".lora_up.scale_weight", None)
                
                groups[base_key] = {
                    "down": down_weight,
                    "up": up_weight,
                    "alpha": alpha,
                    "down_scale": down_scale,
                    "up_scale": up_scale
                }
    elif lora_format == "peft":
        # PEFT format: base_model.model.diffusion_model.blocks.0.self_attn.to_q.lora_A.weight
        keys = list(state_dict.keys())
        for key in keys:
            if key not in state_dict: continue
            # Handle both .lora_A.weight and .lora_A.default.weight patterns
            if ".lora_A.weight" in key or ".lora_A.default.weight" in key:
                if ".lora_A.default.weight" in key:
                    base_key = key.replace(".lora_A.default.weight", "")
                    lora_b_key = base_key + ".lora_B.default.weight"
                else:
                    base_key = key.replace(".lora_A.weight", "")
                    lora_b_key = base_key + ".lora_B.weight"
                
                if lora_b_key not in state_dict: continue
                
                # Map PEFT keys back to model keys
                model_key = base_key
                prefixes_to_strip = ["base_model.model.model.", "base_model.model.", "model."]
                for prefix in prefixes_to_strip:
                    if model_key.startswith(prefix):
                        model_key = model_key[len(prefix):]
                        break
                
                # Special handling for CLIP keys in PEFT
                if "text_encoder" in model_key:
                    # text_encoder.qwen3_4b.transformer.model.layers.0.self_attn.q_proj
                    model_key = model_key.replace("text_encoder.", "")
                elif "qwen3_4b" in model_key:
                    # Handle cases where text_encoder prefix might have been stripped already
                    pass
                
                # If it's a CLIP key, we should NOT apply UNet regex mappings later
                is_clip_key = "qwen3_4b" in model_key or (".layers." in model_key and (".self_attn." in model_key or ".mlp." in model_key))
                
                
                down_weight = state_dict.pop(key)
                up_weight = state_dict.pop(lora_b_key)
                
                alpha_value = None
                alpha_source = None
                for alpha_key in [base_key + ".alpha", base_key + ".lora_alpha", base_key + ".scaling"]:
                    if alpha_key in state_dict:
                        alpha_value = state_dict.pop(alpha_key)
                        alpha_source = alpha_key.split(".")[-1]  # "alpha" / "lora_alpha" / "scaling"
                        break
                
                # Check for INT8 scales
                down_scale = state_dict.pop(base_key + ".lora_A.scale_weight",
                                           state_dict.pop(base_key + ".lora_A.default.scale_weight", None))
                up_scale = state_dict.pop(base_key + ".lora_B.scale_weight",
                                         state_dict.pop(base_key + ".lora_B.default.scale_weight", None))
                
                if is_clip_key:
                    # Try regex mapping for CLIP keys
                    mapped_key = None
                    for pattern, replacement in WAN_LORA_KEY_MAP.items():
                        if "qwen3_4b" in replacement:
                            match = re.match(pattern, model_key)
                            if match:
                                i = match.group(1)
                                mapped_key = replacement.format(i=i)
                                break
                    if mapped_key:
                        model_key = mapped_key

                groups[model_key] = {
                    "down": down_weight,
                    "up": up_weight,
                    "alpha": alpha_value,
                    "alpha_source": alpha_source,
                    "down_scale": down_scale,
                    "up_scale": up_scale,
                    "is_clip": is_clip_key
                }
    elif lora_format == "standard":
        # Standard format: diffusion_model.blocks.0.self_attn.to_q.down.weight
        for key in state_dict:
            if ".down.weight" in key:
                base_key = key.replace(".down.weight", "")
                down_scale = state_dict.get(base_key + ".down.scale_weight", None)
                up_scale = state_dict.get(base_key + ".up.scale_weight", None)
                groups[base_key] = {
                    "down": state_dict[key],
                    "up": state_dict[base_key + ".up.weight"],
                    "alpha": state_dict.get(base_key + ".alpha", None),
                    "down_scale": down_scale,
                    "up_scale": up_scale
                }
            elif ".down_proj.weight" in key:
                base_key = key.replace(".down_proj.weight", "")
                down_scale = state_dict.get(base_key + ".down_proj.scale_weight", None)
                up_scale = state_dict.get(base_key + ".up_proj.scale_weight", None)
                groups[base_key] = {
                    "down": state_dict[key],
                    "up": state_dict[base_key + ".up_proj.weight"],
                    "alpha": state_dict.get(base_key + ".alpha", None),
                    "down_scale": down_scale,
                    "up_scale": up_scale
                }

    # Map groups to model keys
    for base_key, weights in groups.items():
        model_key = None
        
        # Skip regex mapping if already handled (e.g. PEFT CLIP)
        if weights.get("is_clip", False):
            model_key = base_key
        else:
            # Try regex mapping for Kohya-style keys
            for pattern, replacement in WAN_LORA_KEY_MAP.items():
                # Don't apply UNet mappings to CLIP keys
                if "qwen3_4b" in replacement: continue
                
                match = re.match(pattern, base_key)
                if match:
                    i = match.group(1)
                    model_key = replacement.format(i=i)
                    break
        
        # If no regex match, assume the base_key is already the model key (PEFT/Standard)
        if not model_key:
            model_key = base_key
            
        alpha = weights["alpha"]
        alpha_source = weights.get("alpha_source", None)
        
        if alpha is not None:
            if isinstance(alpha, torch.Tensor):
                alpha = float(alpha.item())
            
            # IMPORTANT:
            # - if source is "scaling": it's already alpha/rank, so do NOT divide again
            # - otherwise: treat it as alpha and compute alpha/rank
            if alpha_source == "scaling":
                scale = alpha
            else:
                rank = weights["down"].shape[0]
                if rank == 0:
                    rank = 1
                scale = alpha / rank
        else:
            scale = 1.0
        
        # Get scales for INT8 LoRA
        down_scale = weights.get("down_scale", None)
        up_scale = weights.get("up_scale", None)
        
        parsed_weights.add(
            model_key,
            weights["down"],
            weights["up"],
            scale * strength,
            down_scale=down_scale,
            up_scale=up_scale
        )
    
    # Count INT8 vs float LoRA weights
    int8_count = sum(1 for is_int8 in parsed_weights.is_int8.values() if is_int8)
    float_count = len(parsed_weights.weights) - int8_count
    
    if int8_count > 0:
        print(f"✓ Detected INT8 LoRA: {int8_count} layers quantized, {float_count} layers float")
    
    if debug:
        print(f"\n[DEBUG] Parsed LoRA weights ({len(parsed_weights.weights)} keys):")
        for i, key in enumerate(sorted(parsed_weights.weights.keys())[:20]):  # Show first 20
            is_int8 = parsed_weights.is_int8[key]
            if is_int8:
                down, up, down_scale, up_scale, alpha = parsed_weights.weights[key]
                print(f"  {i+1}. {key} (INT8, rank={down.shape[0]}, in={down.shape[1]}, out={up.shape[0]}, alpha={alpha:.4f})")
            else:
                down, up, alpha = parsed_weights.weights[key]
                print(f"  {i+1}. {key} (float, rank={down.shape[0]}, in={down.shape[1]}, out={up.shape[0]}, alpha={alpha:.4f})")
        if len(parsed_weights.weights) > 20:
            print(f"  ... and {len(parsed_weights.weights) - 20} more")
        
    return parsed_weights
