# Hunyuan Image 3.0 Performance Notes

This document captures performance findings, optimization attempts, and library limitations discovered during development of this ComfyUI node set.

## Hardware Context

Testing performed on:
- **GPU**: NVIDIA RTX 6000 Pro Blackwell (96GB VRAM)
- **System RAM**: 387GB DDR5
- **Storage**: NVMe SSD
- **Model**: Hunyuan Image 3.0 (13B parameters)

## Model Size Reference

| Format | Model Size | VRAM Required |
|--------|-----------|---------------|
| BF16 (full precision) | ~160GB | 160GB+ (doesn't fit in 96GB) |
| INT8 (bitsandbytes) | ~81GB | ~81GB |
| NF4 (bitsandbytes) | ~40GB | ~40GB |

## Performance Benchmarks

### Inference Times (1280x720, 38 steps)

| Configuration | Load Time | Inference Time | Total Cycle |
|--------------|-----------|----------------|-------------|
| INT8, no offload | ~1:40 | ~3:00 | ~4:40 |
| BF16 + smart offload | ~0:38 | ~11:00+ | ~12:00 |
| BF16, no offload | N/A | N/A | Requires 160GB+ VRAM |

### VRAM Requirements for Inference

Empirical formula: `VRAM_inference ≈ 12 × (megapixels)^1.4`

| Resolution | Megapixels | Inference VRAM |
|-----------|------------|----------------|
| 1024×1024 | 1.0 MP | ~12 GB |
| 1280×720 | 0.92 MP | ~11 GB |
| 1920×1080 | 2.07 MP | ~22 GB |
| 2560×1440 | 3.69 MP | ~45 GB |

## Key Findings

### 1. INT8 is Optimal for 96GB GPUs

With 96GB VRAM, INT8 quantization provides the best balance:
- Model fits entirely in VRAM (81GB)
- No PCIe bandwidth bottleneck during inference
- Full GPU utilization
- 3-minute inference vs 11+ minutes with BF16 offloading

### 2. BF16 + Smart Offload is PCIe-Bound

When using `accelerate`'s device_map offloading with BF16:
- ~64GB of weights must shuttle between CPU and GPU
- PCIe 4.0 x16 = ~32 GB/s theoretical, lower in practice
- GPU utilization drops significantly (waiting for data)
- Results in 3.8x slower inference than INT8

### 3. Quality Differences

Trained eyes can notice subtle differences between INT8 and BF16:
- BF16 preserves full model precision
- INT8 has minor quantization artifacts
- For critical/production work, BF16 may be preferred despite time cost

## Optimization Attempts

### Soft Unload (Move Model to CPU RAM)

**Goal**: Keep model in CPU RAM instead of deleting, enabling fast reload (~20-30s) vs disk reload (~1:40).

**Status**: ❌ Not possible with current libraries

#### INT8/NF4 Models (bitsandbytes)
- bitsandbytes does not support `.to()` on quantized models
- Error: `".to" is not supported for "8-bit" bitsandbytes models`
- Quantization state is device-bound
- Would require dequantize → move → requantize path

#### BF16 with Offloading (accelerate)
- accelerate's `device_map` creates meta tensors
- Meta tensors are placeholders with no actual data
- Error: `Cannot copy out of meta tensor; no data!`
- Data lives on disk, loaded on-demand

### Smart Offload Override for INT8

**Goal**: Prevent OOM when generating high-resolution images with INT8.

**Status**: ✅ Implemented

When INT8 model + disabled offload + resolution > 1.2MP:
- Automatically switches to "smart" offload mode
- Prevents inference-time OOM
- Logs warning to user

### Force Unload (Nuclear Option)

**Goal**: Aggressively clear VRAM when normal unload fails.

**Status**: ✅ Implemented

- Multiple GC passes
- Clears all model references
- Resets CUDA memory allocator
- Handles post-OOM memory leaks
- **NEW**: "Nuke Orphaned Tensors" option for stuck memory after OOM

### OOM Memory Leak Issue

**Problem**: When an OOM error occurs during inference:
1. ComfyUI's OOM handler clears Python references to models
2. Our cache reference (`HunyuanModelCache._cached_model`) becomes None
3. BUT the GPU memory is not actually freed - tensors are orphaned
4. Force Unload sees "no model cached" but 70GB+ is stuck in VRAM

**Solution**: The "nuke_orphaned_tensors" option in Force Unload:
1. Scans ALL Python objects via `gc.get_objects()`
2. Finds any `torch.nn.Module` instances and clears their parameters
3. Finds any raw `torch.Tensor` on CUDA and replaces with empty CPU tensors
4. Forces multiple GC passes to release the memory

**Usage**: After an OOM error, run Force Unload with `nuke_orphaned_tensors=True`.

## Library Limitations Summary

### bitsandbytes Limitations

| Feature | Status | Impact |
|---------|--------|--------|
| `.to()` device movement | ❌ Not supported | Cannot soft unload INT8/NF4 models |
| CPU inference | ❌ Not supported | Model must stay on GPU |
| Re-quantization | ❌ Not exposed | No path to move and re-quantize |

### accelerate Limitations

| Feature | Status | Impact |
|---------|--------|--------|
| Meta tensor materialization | ❌ Not available | Cannot move offloaded models |
| Pinned memory for offload | ⚠️ Partial | Some speedup possible |
| Selective layer pinning | ❌ Not granular | All-or-nothing offloading |

## Recommendations by Use Case

### Iterative/Experimental Work
- Use **INT8** loader
- Accept 1:40 reload time between sessions
- Fastest inference (3 min)

### Quality-Critical/Production Work
- Use **BF16 Full Loader** with smart offload
- Accept 11+ min inference time
- Best quality output

### Multi-Model Workflows
- Use **Force Unload** between models
- Ensures clean VRAM state
- Prevents memory leak accumulation

### High-Resolution Generation
- Use **Budget loaders** with appropriate headroom
- Let auto-offload manage VRAM
- Or manually set offload_mode="smart"

## Future Improvements

Potential improvements pending library updates:

1. **bitsandbytes**: Device movement for quantized models
   - See: [Feature Request Link TBD]
   
2. **accelerate**: Meta tensor materialization
   - See: [Feature Request Link TBD]

3. **Alternative quantization**: GGUF-style quantization supports CPU↔GPU movement but lacks HuggingFace pipeline integration for image models.

## Version History

- **2025-11-30**: Initial documentation based on RTX 6000 Pro testing
- Soft unload investigation completed
- Force unload implemented
- Auto-offload override for INT8 implemented
