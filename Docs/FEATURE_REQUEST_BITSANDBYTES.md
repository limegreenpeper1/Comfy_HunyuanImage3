# Feature Request: Support `.to()` Device Movement for Quantized Models

## Summary

Enable moving quantized models (8-bit and 4-bit) between devices using `.to()`, allowing GPU→CPU→GPU movement without reloading from disk.

## Motivation

### Use Case: Fast Model Swapping in Multi-Model Workflows

When running inference pipelines that use multiple large models sequentially (e.g., image generation → upscaling → face restoration), users need to:

1. Run Model A on GPU
2. Free GPU VRAM for Model B
3. Run Model B on GPU
4. Restore Model A for next iteration

**Current behavior**: 
- Calling `.to('cpu')` or `.to('cuda')` on a quantized model raises:
  ```
  ".to" is not supported for "8-bit" bitsandbytes models
  ```
- Users must delete the model and reload from disk (slow, 1-2+ minutes for large models)

**Desired behavior**:
- Move quantized model to CPU RAM (fast, ~10-20 seconds)
- Model stays in system RAM while other models use GPU
- Restore to GPU when needed (fast, ~10-20 seconds)

### Real-World Example

Testing with Hunyuan Image 3.0 (13B parameters, ~81GB INT8):
- **Reload from NVMe SSD**: ~1:40 minutes
- **Theoretical CPU→GPU transfer** (if supported): ~10-20 seconds at DDR5 speeds

For iterative workflows generating 10+ images with model switching, this adds 15+ minutes of unnecessary I/O overhead.

## Technical Context

### Current Limitation

The quantization state (`quant_state`) appears to be tied to the original device. The `Linear8bitLt` and `Linear4bit` modules explicitly block `.to()` operations.

### Suggested Approaches

1. **Dequantize → Move → Requantize**
   - Store original quantization config
   - Dequantize weights to full precision on source device
   - Move dequantized weights to target device
   - Requantize on target device
   - Higher memory peak during move, but enables the workflow

2. **Direct State Movement**
   - Move the underlying tensors and `quant_state` together
   - Update device references in `quant_state`
   - May require CUDA↔CPU implementations for quantization ops

3. **Lazy Re-quantization**
   - Move dequantized weights to CPU
   - Store quantization config
   - Re-quantize on-demand when `.to('cuda')` is called

## Environment

- **bitsandbytes version**: 0.44.x / 0.45.x
- **PyTorch**: 2.0+
- **Hardware**: Multi-GPU and high-RAM systems where CPU parking is viable

## Workaround Attempts

We attempted to:
1. Manually iterate through modules and move non-quantized components
2. Clear CUDA cache and use the model from CPU
3. Save/reload from a RAM disk

None provide the seamless experience of true `.to()` support.

## Impact

This would benefit:
- **ComfyUI/Automatic1111 users**: Multi-model workflows are common
- **Production inference pipelines**: Model orchestration with limited VRAM
- **Research workflows**: Comparing quantized vs full-precision models

## Related Issues

- [Link to any existing issues if found]

## Proposed API

```python
# Current (fails)
model = model.to('cpu')  # Raises error

# Desired
model = model.to('cpu')  # Moves model to CPU, preserves quantization state
model = model.to('cuda:0')  # Moves back, ready for inference

# Alternative explicit API
model = model.dequantize_and_move('cpu')  # If implicit .to() is too complex
model = model.move_and_requantize('cuda:0')
```

## Willingness to Contribute

I'm happy to help test any proposed implementation or provide additional use case details.

---

**Thank you for considering this feature request!**
