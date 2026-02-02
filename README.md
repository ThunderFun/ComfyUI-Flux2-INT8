# ComfyUI-Wan-INT8

A ComfyUI custom node for fast INT8 quantized inference of Wan video generation models and other diffusion models.

This is a fork of the original Flux INT8 Acceleration project with additional features including extra Triton kernels, CPU offloading, and enhanced LoRA support.

> ⚠️ **Disclaimer: This is experimental software and may be unstable or buggy. Features may not work as expected, and breaking changes may occur. Use at your own risk.**

## Features

### W8A8 Triton Kernel
Custom W8A8 (8-bit weights, 8-bit activations) Triton kernel for optimized INT8 inference. The kernel performs fused quantization, matrix multiplication, and dequantization for improved performance on supported GPUs.

**GPU Architecture Support:**
- **Ampere** (RTX 30xx, A100) - Optimized autotuned kernel with `tl.make_block_ptr` pipelining
- **Ada** (RTX 40xx) - Uses Ampere kernel with Ada-friendly configs
- **Turing** (RTX 20xx) - Fallback kernel with safe configurations
- **Older GPUs** - Fixed fallback kernel

The kernel automatically selects the best implementation based on your GPU's compute capability.

### Supported Models

#### WAN 2.2 INT8 Models
The following pre-quantized models are available and functional:

- [wan2.2_i2v_A14b_high_noise_int8_lightx2v_4step_1030.safetensors](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)
- [wan2.2_i2v_A14b_low_noise_int8_lightx2v_4step.safetensors](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)

#### Model Compatibility
| Model | Status | Notes |
|-------|--------|-------|
| Wan 2.2 | ✅ Tested | Full support with LoRA |
| Flux 2 | ✅ Supported | Excludes embedding layers from quantization |
| Z-Image | ✅ Supported | Includes Diffusers format conversion |

### INT8 Quantization
QuIP quantization is recommended because it typically offers high precision and tends to work well with LoRA adapters, primarily when applied to transformer-based models. Use the [convert_to_quant_QuIP_INT8](https://github.com/ThunderFun/convert_to_quant_QuIP_INT8) project to quantize your models.

### CPU Offloading (Optional)
Optional CPU offloading can be enabled via node settings. When enabled, float LoRA weights are kept on CPU and moved to GPU during inference to save VRAM. **Note:** This feature may exhibit bugs when enabled.

### LoRA Support
- Custom LoRA loader nodes for INT8 models
- Supports both float and INT8 quantized LoRAs
- Memory-efficient chunked forward pass
- Optional CPU offloading for float LoRA weights
- LoRA weight caching for improved performance

**Note:** LoRA functionality may not be stable and could exhibit issues depending on the specific model and LoRA combination.

### Hadamard-QuIP Kernel (Experimental)
The Hadamard-QuIP kernel exists but is **disabled by default** due to high VRAM requirements. It may also be very buggy and unusable. Enable with environment variable `INT8_HADAMARD_QUIP=1` or use `set_hadamard_quip_enabled(True)` at runtime.

## Requirements

- Working ComfyUI installation (latest comfy recommended)
- PyTorch with CUDA support (cu130 recommended)
- Triton (`pip install triton`)
- Windows untested, but triton-windows may work

## Installation

1. Clone this repository into your ComfyUI custom nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ThunderFun/ComfyUI-Wan-INT8.git
```

2. Install dependencies:
```bash
pip install triton
```

## Available Nodes

### WanVideoINT8Loader
Loads INT8 tensorwise quantized diffusion models with fast torch._int_mm inference.

**Inputs:**
- `unet_name`: Select from available diffusion models
- `model_type`: Choose from `wan2.2`, `wan2.1`, `flux2`, or `z-image`
- `offload_to_cpu`: Enable/disable CPU offloading for float weights
- `auto_convert_to_int8`: Automatically convert non-INT8 weights to INT8
- `debug_mode`: Enable detailed debug output

### WanLoRALoaderINT8
Loads and applies LoRA weights to INT8 models.

**Inputs:**
- `model`: The model to apply LoRA to
- `lora_name`: Select from available LoRA files
- `strength`: LoRA strength (-2.0 to 2.0)
- `offload_to_cpu`: Enable CPU offloading for float LoRA weights
- `debug`: Enable debug output

### WanLoRALoaderWithCLIPINT8
Same as WanLoRALoaderINT8 but with CLIP support for conditioned models.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INT8_HADAMARD_QUIP` | `0` | Enable Hadamard-QuIP kernel |
| `INT8_DISABLE_HADAMARD_QUIP` | ` ` | Force disable Hadamard-QuIP |
| `INT8_DEBUG_MODE` | `0` | Enable debug logging |
| `INT8_ENABLE_CUDA_SYNC` | `0` | Enable CUDA synchronization for debugging |
| `INT8_CLEAR_CACHE` | `auto` | CUDA cache clearing strategy (`always`/`never`/`auto`) |
| `INT8_LORA_CACHE_SIZE` | `32` | Maximum cached LoRA patches |
| `INT8_LOG_KERNEL` | `0` | Log kernel selection per layer |
| `INT8_FORCE_KERNEL` | ` ` | Force kernel (`ampere`/`fallback`/`fixed`) |
| `HADAMARD_CHUNK_SIZE` | `2048` | Chunk size for Hadamard transform |
| `HADAMARD_DIAGNOSTICS` | `0` | Enable Hadamard diagnostic output |

## Runtime Functions

```python
from int8_quant import set_hadamard_quip_enabled, is_hadamard_quip_enabled, print_kernel_summary, reset_kernel_stats

# Enable/disable Hadamard-QuIP kernel
set_hadamard_quip_enabled(True)

# Check current status
print(is_hadamard_quip_enabled())

# Print kernel usage summary
print_kernel_summary()

# Reset kernel statistics
reset_kernel_stats()
```

## Workflow

See [Workflow.png](Workflow.png) for an example ComfyUI workflow setup.

## Performance Notes

- This node is unlikely to be faster than proper FP8 on 40-Series and above GPUs
- Works best with torch.compile enabled for full speedup
- LoRA support may add overhead depending on rank and number of patches

## Troubleshooting

### Out of Memory Errors
- Enable CPU offloading for float LoRA weights
- Reduce batch size or resolution
- Use `INT8_CLEAR_CACHE=always` to aggressively clear CUDA cache

### Numerical Issues
- Enable debug mode to check for NaN/Inf values
- Verify model was quantized correctly with recommended tools
- Check weight scales are valid (non-zero, finite values)

### LoRA Not Applying
- Verify LoRA dimensions match the target model
- Check debug output for dimension mismatch warnings

## Credits

### Fork Maintainer
- **ThunderFun** - Current fork maintainer

### Original Code
- **BobJohnson24** - Original code this fork is based on

### Contributions
- dxqb for the INT8 code: [Nerogar/OneTrainer#1034](https://github.com/Nerogar/OneTrainer/pull/1034)
- silveroxides for the base conversion code: [silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant)
- silveroxides for showing how to register new data types to Comfy: [silveroxides/ComfyUI-QuantOps](https://github.com/silveroxides/ComfyUI-QuantOps)

## License

See [LICENSE](LICENSE) for license information.
