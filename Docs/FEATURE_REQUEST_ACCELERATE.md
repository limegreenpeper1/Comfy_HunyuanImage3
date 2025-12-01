# Feature Request: Add API to Materialize Meta Tensors for Device Movement

## Summary

Provide an API to materialize all meta tensors in a model loaded with `device_map`, enabling subsequent `.to()` device movement.

## Motivation

### Use Case: Fast Model Swapping with CPU RAM Parking

When using `device_map="auto"` or custom device maps for large models that exceed GPU VRAM, `accelerate` creates meta tensors as placeholders. This is excellent for memory efficiency during inference.

However, users with large system RAM (256GB+) want to:

1. Load model with device_map (fits in available VRAM)
2. Run inference
3. Move entire model to CPU RAM (freeing GPU for other models)
4. Later, move model back to GPU (faster than disk reload)

This "CPU parking" pattern is common in multi-model workflows.

### Current Limitation

When calling `.to()` or `.cpu()` on a model with meta tensors:

```
RuntimeError: Cannot copy out of meta tensor; no data!
```

The meta tensors have no actual data - they're placeholders that get populated on-demand from disk/CPU.

### Real-World Example

Testing with Hunyuan Image 3.0 (25B parameters, ~160GB BF16):

- GPU: 96GB VRAM (model doesn't fit without offloading)
- System RAM: 387GB (can easily hold the full model)
- Disk reload time: ~2-3 minutes
- Theoretical RAM→GPU time: ~20-30 seconds

**Current workflow**: Delete model, reload from disk each time

**Desired workflow**: Park on CPU RAM, restore quickly

## Proposed API

```python
from accelerate import dispatch_model, materialize_meta_tensors

# Load with device_map (creates meta tensors for offloaded parts)
model = AutoModel.from_pretrained("large-model", device_map="auto")

# Run inference...
output = model(input)

# NEW: Materialize all meta tensors (load from disk to CPU)
model = materialize_meta_tensors(model, target_device="cpu")

# Now model has no meta tensors, can move freely
model = model.to("cpu")  # Works!

# ... use GPU for other models ...

# Restore to GPU
model = model.to("cuda:0")  # Works because no meta tensors
```

### Alternative API Options

```python
# Option 1: Model method
model.materialize_all(device="cpu")

# Option 2: Context manager
with model.materialized():
    model = model.cpu()
    # ... do other stuff ...
    model = model.cuda()

# Option 3: Dispatch option
model = dispatch_model(model, device_map="auto", allow_materialize=True)
model.materialize()  # Loads everything to CPU
```

## Technical Considerations

1. **Memory Impact**: Materializing requires enough RAM to hold the full model. This should be documented/warned.

2. **Backward Compatibility**: This is additive - existing code continues to work with meta tensors.

3. **Performance**: One-time disk I/O during materialize, then fast RAM operations afterward.

4. **Hooks Cleanup**: The `AlignDevicesHook` and other dispatch hooks may need adjustment after materialization.

## Environment

- **accelerate version**: 0.25.x+
- **PyTorch**: 2.0+
- **Use case**: High-RAM systems (256GB+) with limited VRAM

## Workaround Attempts

We tried:

1. Manually iterating parameters and calling `.to()` - fails on meta tensors
2. Using `torch.load` with mmap - doesn't integrate with HuggingFace pipelines
3. Checking for meta device before move - can detect but not resolve

## Impact

This would benefit:

- **Multi-model inference pipelines**: Common in image/video generation
- **A/B testing workflows**: Compare models without repeated disk I/O  
- **Production systems**: Reduce latency in model orchestration
- **Interactive applications**: Faster model switching for user-facing tools

## Related Context

- Meta tensors were introduced for memory efficiency
- The inverse operation (materialize for flexibility) would complete the feature set
- Similar to how `torch.Tensor.contiguous()` materializes views

## Willingness to Contribute

Happy to help test implementations or provide additional use case details.

---

**Thank you for considering this feature request!**
