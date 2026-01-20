# Flux2 INT8 Acceleration

This node speeds up Flux2 in ComfyUI by using INT8 quantization, delivering ~2x faster inference on my 3090, but it should work on any NVIDIA GPU with enough INT8 TOPS. It's unlikely to be faster than proper FP8 on 40-Series and above. 
Works with lora, torch compile (needed to get full speedup).

Requirements:
Working ComfyKitchen (needs latest comfy and possibly pytorch with cu130)
Triton

Windows untested, but I hear triton-windows exists.

# Credits:

## dxqb for the *entirety* of the INT8 code, it would have been impossible without them:
https://github.com/Nerogar/OneTrainer/pull/1034

If you have a 30-Series GPU, OneTrainer is also the fastest current lora trainer thanks to this. Please go check them out!!


# silveroxides for providing a base to hack the INT8 conversion code onto.
https://github.com/silveroxides/convert_to_quant

# Also silveroxides for showing how to properly register new data types to comfy
https://github.com/silveroxides/ComfyUI-QuantOps

# The unholy trinity of AI slopsters I used to glue all this together over the course of a day
