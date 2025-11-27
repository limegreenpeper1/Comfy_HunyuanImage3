# INT8 Model Loading - Technical Notes

## Session Date: November 25, 2025

---

## Key Learnings

### 0. New Budget-Friendly Nodes

To keep legacy nodes untouched, all GPU budget controls now live in dedicated nodes:

- **Loader:** `Hunyuan 3 Loader (INT8 Budget)` exposes the GPU budget slider and records reserve metadata.
- **Loader (NF4):** `Hunyuan 3 Loader (NF4 Low VRAM+)` mirrors the same behavior for NF4 checkpoints.
- **Generators:** `Hunyuan 3 Generate (Large Budget)` and `Hunyuan 3 Generate (Low VRAM Budget)` understand the loader metadata, implement smarter "smart" mode checks, and emit MemoryTracker telemetry in the node status.

Use these new nodes whenever you want budget sliders or telemetry output; existing nodes behave exactly as before this change.

### 1. INT8 Model Variants

There are **TWO** different INT8 quantization variants for HunyuanImage-3:

#### **INT8-Full** (60GB on disk → 96GB loaded)
- Everything quantized to INT8 on disk
- During loading, critical layers are **dequantized** back to bfloat16
- Final size: ~96GB (60GB INT8 weights + 36GB dequantized critical layers)
- Metadata: `"quantization_method": "bitsandbytes_int8_full"`

#### **INT8-Selective** (96GB on disk → 130GB loaded)
- Only transformer weights quantized to INT8
- VAE, attention, and critical layers already stored as bfloat16 on disk
- Final size: ~130GB (96GB on disk + 34GB loading overhead)
- Metadata: `"quantization_method": "bitsandbytes_int8"` (not "full")

**CRITICAL**: Both variants end up with the same critical layers in bfloat16 after loading. The difference is whether we're dequantizing (Full) or preserving (Selective).

---

### 2. Bitsandbytes Configuration Issues

#### **The `llm_int8_enable_fp32_cpu_offload` Conflict**

**Problem**: 
```python
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,  # ❌ CONFLICTS with device_map="auto"
)
```

When using `device_map="auto"` with `max_memory` constraints, bitsandbytes validation fails with:
```
ValueError: Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM...
```

**Solution**:
```python
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=skip_modules,
    llm_int8_enable_fp32_cpu_offload=False,  # ✅ Let device_map handle offload
)

load_kwargs = dict(
    device_map="auto",
    max_memory={0: target_vram, "cpu": "100GiB"},  # We control offload
    quantization_config=quant_config,
)
```

**Why**: `llm_int8_enable_fp32_cpu_offload=True` is incompatible with `device_map="auto"`. We manage CPU offload ourselves via `max_memory` constraints.

---

### 3. Critical Layers That MUST Stay Full Precision

The following layers cause severe corruption (colorful noise artifacts) if quantized to INT8:

```python
skip_modules = [
    # VAE/Autoencoder (image encoding/decoding)
    "vae", "model.vae", "vae.decoder", "vae.encoder",
    "autoencoder", "model.autoencoder", "autoencoder.decoder", "autoencoder.encoder",
    
    # Vision components
    "vision_model", "model.vision_model",
    "vision_aligner", "model.vision_aligner",
    
    # Embeddings and core layers
    "timestep_emb", "model.timestep_emb",
    "patch_embed", "model.patch_embed",
    "time_embed", "model.time_embed",
    "time_embed_2", "model.time_embed_2",
    "final_layer", "model.final_layer",
    
    # Language model components
    "model.wte", "wte",
    "model.ln_f", "ln_f",
    "lm_head", "model.lm_head",
    
    # Attention layers
    "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
    "attn.qkv_proj", "attn.out_proj",
    "self_attn", "cross_attn",
]
```

**Important**: These are **pattern matches** - any layer name containing these strings will be skipped.

---

### 4. Memory Management Strategy

#### **Target: 55% GPU Usage**

For a 96GB GPU:
- **Target**: ~52GB on GPU (55% of total)
- **Offloaded**: ~44GB to CPU (45% of model)
- **Reserved**: ~43GB free VRAM for inference pulsing

```python
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info(0)
    target_model_vram = int(total * 0.55)  # Use 55% for model
    max_memory = {
        0: target_model_vram,
        "cpu": "100GiB"
    }
```

#### **Why 55%?**
- 96GB model on 96GB GPU leaves no headroom
- Inference causes temporary VRAM spikes (up to 12GB per megapixel)
- Without headroom: 99% VRAM usage → stalls/OOM
- With 55% target: ~43GB free → smooth inference

#### **GPU Budget Slider (Loaders & Generators)**

- The budget slider now lives in `Hunyuan 3 Loader (NF4 Low VRAM+)` and `Hunyuan 3 Loader (INT8 Budget)`.
- Set this to the VRAM you are comfortable dedicating to weights; anything beyond is auto-offloaded to CPU RAM.
- Suggested starting points:
    - 24GB GPUs → **18‑20GB** target
    - 32GB GPUs → **26‑28GB** target
    - 96GB GPUs → **52GB** (default)
- Large/offload generators (`Hunyuan 3 Generate (Large Budget)` / `Low VRAM Budget`) also include an optional per-run override so you can experiment with different budgets without reloading the model.

---

### 5. Smart Mode Detection (Fixed)

**Old "Stupid" Logic**: Tried to detect model type and predict requirements
**New "Actually Smart" Logic**: Just check current VRAM state

```python
if offload_mode == "smart" and torch.cuda.is_available():
    megapixels = (width * height) / 1_000_000
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / 1024**3
    allocated_gb = (total_bytes - free_bytes) / 1024**3
    percent_used = (allocated_gb / total_gb) * 100
    
    loader_reserve_gb = getattr(model, '_loader_reserve_gb', 0)
    inference_requirement = (megapixels * 12.0) + 5.0 + loader_reserve_gb
    
    # Three simple rules:
    if free_gb < 25.0:  # Less than 25GB free (allows 5090/4090 to skip)
        should_offload = True
    elif percent_used > 70.0:  # Using more than 70% of GPU
        should_offload = True
    elif free_gb < inference_requirement:  # Not enough for inference
        should_offload = True
```

**Key Insight**: Model is already loaded - don't try to predict model size, just check if there's enough room for inference.

---

### 6. Terminology Clarification

#### **"Keep in bfloat16" vs "Dequantize to bfloat16"**

**Wrong terminology** (what we kept saying):

- "Keep VAE in bfloat16"

**Correct terminology**:

- **INT8-Full**: "Dequantize/expand VAE from INT8 → bfloat16"
- **INT8-Selective**: "Preserve VAE as bfloat16 (already full precision)"

The `llm_int8_skip_modules` parameter does **different things** depending on what's on disk:

1. If layer is INT8 on disk → causes **dequantization** to bfloat16
2. If layer is bfloat16 on disk → **prevents quantization**, keeps as-is

Same parameter, different behavior based on checkpoint content.

---

### 7. Performance Expectations

**Target**: INT8 should be **2-3x faster** than BF16

- BF16: ~6 minutes for 1MP image
- INT8 Expected: ~2-3 minutes
- INT8 Reality (before fixes): 12 minutes (2x SLOWER!)

**Why it was slow**:

- Loading 81GB to GPU (85% usage)
- During inference: 99% VRAM → constant swapping/stalls
- OOM after a few minutes

**After fixes** (expected):

- Loading ~52GB to GPU (55% usage)
- During inference: ~70-80% VRAM (room to pulse)
- Should hit 2-3x speedup target

---

### 8. Debug Verification Added

Added logging to verify which layers are actually in which dtype:

```python
logger.info("Verifying layer dtypes after load:")
dtype_summary = {}
for name, param in model.named_parameters():
    dtype = str(param.dtype)
    if any(key in name.lower() for key in ['vae', 'autoencoder', 'vision', 'embed', 'final', 'lm_head', 'wte', 'ln_f']):
        dtype_summary[dtype].append(name)

for dtype, layers in dtype_summary.items():
    logger.info(f"  {dtype}: {len(layers)} layers")
    logger.info(f"    Examples: {', '.join(layers[:3])}")
```

This shows if skip_modules patterns are actually matching the real layer names.

---

### 9. Common Errors & Solutions

#### **Error**: "Cannot copy out of meta tensor"

**Cause**: Model already has meta tensors, trying to apply cpu_offload() again
**Solution**: Check for `has_meta` flag before applying cpu_offload

#### **Error**: "Some modules are dispatched on the CPU..."

**Cause**: `llm_int8_enable_fp32_cpu_offload=True` with `device_map="auto"`
**Solution**: Set to `False`, manage offload via max_memory

#### **Error**: VAE corruption (colorful noise)

**Cause**: VAE layers still quantized to INT8
**Solution**: Verify skip_modules patterns match actual layer names, check dtype verification log

#### **Error**: 99% VRAM usage, OOM

**Cause**: Model loaded too close to GPU limit (81GB on 96GB GPU)
**Solution**: Use max_memory to target 55% GPU usage

---

### 10. Telemetry & Status Output

- All generate nodes (standard, large/offload, low VRAM) now track **peak GPU VRAM** and **system RAM** for every run.
- The values are logged and appended to the node's status string, e.g. `... | Peak VRAM 27.8GB | Peak RAM 42.1GB`.
- Use those stats to tune the GPU budget slider—keep peaks ~10GB under your physical limit to leave room for inference spikes.

---

## Optimal Device Map (Manual Configuration)

If you want to manually control layer placement instead of using `device_map="auto"`, here's a recommended optimal mapping for 96GB GPU:

```python
optimal_device_map = {
    # Critical layers on GPU (must be fast)
    "vision_model": 0,
    "vision_aligner": 0,
    "timestep_emb": 0,
    "patch_embed": 0,
    "time_embed": 0,
    "final_layer": 0,
    "time_embed_2": 0,
    "model.wte": 0,
    "model.ln_f": 0,
    "lm_head": 0,
    
    # First 6 transformer layers on GPU
    "model.layers.0": 0,
    "model.layers.1": 0,
    "model.layers.2": 0,
    "model.layers.3": 0,
    "model.layers.4": 0,
    "model.layers.5": 0,
    
    # Remaining layers offloaded to CPU
    "model.layers.6": "cpu",
    "model.layers.7": "cpu",
    "model.layers.8": "cpu",
    "model.layers.9": "cpu",
    "model.layers.10": "cpu",
    "model.layers.11": "cpu",
    "model.layers.12": "cpu",
    "model.layers.13": "cpu",
    "model.layers.14": "cpu",
    "model.layers.15": "cpu",
    "model.layers.16": "cpu",
    "model.layers.17": "cpu",
    "model.layers.18": "cpu",
    "model.layers.19": "cpu",
    "model.layers.20": "cpu",
    "model.layers.21": "cpu",
    "model.layers.22": "cpu",
    "model.layers.23": "cpu",
    "model.layers.25": "cpu",
    "model.layers.26": "cpu",
    "model.layers.27": "cpu",
    "model.layers.28": "cpu",
    "model.layers.29": "cpu",
    "model.layers.30": "cpu",
    "model.layers.31": "cpu",
    "model.layers.32": "cpu",
}
```

**Note**: This keeps critical components and first 6 transformer layers on GPU, offloading the bulk of transformer layers to CPU. This provides good balance between speed and VRAM usage.

---

## Code Changes Summary

### File: `hunyuan_quantized_nodes.py`

1. **Lines 773-831**: Unified skip_modules list for both variants
2. **Lines 833-837**: Disabled `llm_int8_enable_fp32_cpu_offload` (set to False)
3. **Lines 852-875**: INT8-Full uses 55% GPU target with max_memory
4. **Lines 876-898**: INT8-Selective uses 55% GPU target with max_memory
5. **Lines 912-928**: Added dtype verification logging
6. **Lines 1623-1650**: Simplified smart mode detection (removed model size prediction)

---

## Testing Checklist

- [ ] INT8-Full loads to ~52GB (not 81GB)
- [ ] INT8-Selective loads without bitsandbytes validation error
- [ ] No VAE corruption (colorful noise) in generated images
- [ ] VRAM stays below 85% during inference (not 99%)
- [ ] Performance is 2-3x faster than BF16 (not slower)
- [ ] Smart mode correctly detects when offload is needed
- [ ] Dtype verification log shows VAE/critical layers in bfloat16

---

## Notes for Future

- Both INT8 variants have similar final memory footprint (~96-130GB)
- The 55% GPU target is optimized for 96GB GPU - may need adjustment for other GPU sizes
- Smart mode threshold is 25GB free (works for 5090/4090, adjust if needed)
- Skip_modules uses pattern matching - if corruption occurs, check actual layer names with dtype verification log
