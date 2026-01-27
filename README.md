# FORK EDITS
### INT8 Support:
  - The **WAN 2.2** INT8 files are functional.
  
  - [wan2.2_i2v_A14b_high_noise_int8_lightx2v_4step_1030.safetensors](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)
  
  - [wan2.2_i2v_A14b_low_noise_int8_lightx2v_4step.safetensors](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)
  
  QuIP quantization is recommended because it typically offers high precision and tends to work well with LoRA adapters, primarily when applied to transformer‑based models: https://github.com/ThunderFun/convert_to_quant_QuIP_INT8

### CPU Offloading (Always‑On, Buggy):
  Added offloading to the CPU, but it cannot currently be disabled and may exhibit bugs.

### LoRA Support:
- Introduced a custom LoRA node.  
- Not thoroughly tested; it may work well in some cases and not in others, depending on how the LoRA was trained or on the precision of the INT8 model.
- Supports LoRAs quantized to INT8, though it incurs higher memory usage.

# Flux2 INT8 Acceleration

This node speeds up Flux2 in ComfyUI by using INT8 quantization, delivering ~2x faster inference on my 3090, but it should work on any NVIDIA GPU with enough INT8 TOPS. It's unlikely to be faster than proper FP8 on 40-Series and above. 
Works with lora, torch compile (needed to get full speedup).

We auto-convert flux2 klein to INT8 on load if needed. Pre-quantized checkpoints with slightly higher quality and enabling faster loading are available here: 
https://huggingface.co/bertbobson/FLUX.2-klein-9B-INT8-Comfy

# Metrics:

Measured at 1024x1024, 26 steps with Flux2 Klein Base 9B.

| Format | Speed (s/it) | Relative Speedup |
|-------|--------------|------------------|
| bf16 | 2.07 | 1.00× |
| bf16 compile | 2.24 | 0.92× |
| fp8 | 2.06 | 1.00× |
| int8 | 1.64 | 1.26× |
| int8 compile | 1.04 | 1.99× |
| gguf8_0 compile | 2.03 | 1.02× |



# Requirements:
Working ComfyKitchen (needs latest comfy and possibly pytorch with cu130)

Triton

Windows untested, but I hear triton-windows exists.

# Credits:

## dxqb for the *entirety* of the INT8 code, it would have been impossible without them:
https://github.com/Nerogar/OneTrainer/pull/1034

If you have a 30-Series GPU, OneTrainer is also the fastest current lora trainer thanks to this. Please go check them out!!


## silveroxides for providing a base to hack the INT8 conversion code onto.
https://github.com/silveroxides/convert_to_quant

## Also silveroxides for showing how to properly register new data types to comfy
https://github.com/silveroxides/ComfyUI-QuantOps

## The unholy trinity of AI slopsters I used to glue all this together over the course of a day
