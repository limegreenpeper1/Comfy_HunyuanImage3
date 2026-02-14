# Image Base Size Override — Exploration Notes

**Date:** February 2026  
**Status:** Experimental feature tested and **reverted** — model does not respect non-default base sizes.

## Summary

We attempted to expose `image_base_size` as a configurable parameter on the Instruct Loader, allowing users to generate at higher resolutions (1536, 2048, 3072, 4096). The implementation was technically correct but the model ignores the override.

## Background

### Tokenizer Vocabulary Evidence

The model's tokenizer (`tokenization_hunyuan_image_3.py`) contains **10 size tokens**:

| Token | ID | Base Size |
|---|---|---|
| `<img_size_256>` | 128034 | 256 |
| `<img_size_512>` | 128035 | 512 |
| `<img_size_768>` | 128036 | 768 |
| `<img_size_1024>` | 128037 | 1024 |
| `<img_size_1536>` | 128038 | 1536 |
| `<img_size_2048>` | 128039 | 2048 |
| `<img_size_3072>` | 128040 | 3072 |
| `<img_size_4096>` | 128041 | 4096 |
| `<img_size_6144>` | 128042 | 6144 |
| `<img_size_8192>` | 128043 | 8192 |

These tokens are embedded in the `img_ratio` bot task via `size_token_id()` at line ~1431 of `modeling_hunyuan_image_3.py`. This strongly suggests multi-resolution support was **architecturally designed in** but likely **not trained**.

### ResolutionGroup

The `ResolutionGroup` class generates aspect-ratio buckets:
- Range: `base_size // 2` to `base_size * 2` (in both height and width)
- Step: `base_size // 16`
- Align: 16 (VAE downsample factor)
- Generates ~33 valid (H, W) buckets for any given base_size

## What We Implemented

1. **Dynamic preset generation**: `_build_resolution_presets(base_size)` generated resolution dropdown entries for any base_size
2. **Loader parameter**: `image_base_size` dropdown with values [1024, 1536, 2048, 3072, 4096]
3. **ResolutionGroup rebuild**: After model load, rebuilt `model.image_processor.vae_reso_group` with new base_size
4. **Config update**: Set `model.config.image_base_size` to the new value
5. **Downstream support**: `parse_resolution()` accepted model parameter to use model-attached presets

The implementation was verified working — logs confirmed new buckets were generated correctly.

## Test Results

### base_size=2048 with INT8 (bitsandbytes)
- **Result: CRASH** — `igemmlt` CUDA kernel error at line 438 in `ops.cu`
- **Cause**: 2048×2048 = 4MP → 128×128 = 16,384 image tokens. The INT8 GEMM kernel's CUDA grid dimensions can't handle sequences this long
- **Note**: NF4 or BF16 might work since they don't use the same INT8 kernel

### base_size=1536 with INT8 (bitsandbytes)
- **Result: Runs but output is still ~1MP** — e.g., 919×1141 instead of expected ~1536×1536
- **Observation**: Portrait input sometimes produced landscape output (aspect ratio flip)

### base_size=2048 with T2I (text-to-image)
- **Result: Output was 1024×768** — the override was completely ignored

## Root Cause Analysis

The model's output resolution is determined by a **ratio prediction** step:
1. `SliceVocabLogitsProcessor` constrains generation to `<img_ratio_0>` through `<img_ratio_36>` tokens
2. The model predicts one of these ratio indices
3. The predicted ratio is looked up in `vae_reso_group.data[ratio_index]` to get (H, W)
4. The diffusion pipeline generates at that (H, W)

Even though we rebuilt `vae_reso_group` with larger buckets, the model's **ratio prediction behavior was trained only at base_size=1024**. The model's internal weights predict ratio indices calibrated for 1024-based buckets, so:
- The ratio index → resolution mapping changes, but the model still predicts the same distribution of indices
- The actual pixel generation (diffusion head, VAE decoder) was likely only trained at 1024-scale
- The `<img_size_N>` tokens may be architectural placeholders that were never backpropagated through

## What Would Be Needed

For multi-resolution to actually work, likely need:
1. **Fine-tuning** at higher base sizes so the diffusion head learns to generate at those resolutions
2. **Or** Tencent releasing models explicitly trained at higher base sizes (the tokenizer infrastructure is ready)
3. **Or** a post-hoc upscaling approach (tile-based diffusion, SDEdit refinement, etc.)

## INT8 Kernel Limitation

Even if the model supported higher resolutions, bitsandbytes INT8 has a hard limit:
- The `igemmlt` kernel uses CUDA grid dimensions that overflow at ~16K+ sequence length
- base_size=2048 → 16,384 image tokens → exceeds kernel grid limits
- Workarounds: Use NF4 (4-bit, no GEMM kernel) or BF16 (native float ops)

## Files Modified (Now Reverted)

All changes were in `hunyuan_instruct_nodes.py`:
1. Resolution presets section (lines ~112-181) — dynamic generation
2. `parse_resolution()` — model parameter for custom presets
3. `INPUT_TYPES` — `image_base_size` dropdown
4. `load_model()` — parameter and parsing
5. Post-load block — ResolutionGroup rebuild + model metadata
6. Generate node — custom parse_resolution call + logging
7. ImageEdit node — base_size logging
8. MultiFusion node — custom parse_resolution call + logging

## Key Upstream Code References

- `modeling_hunyuan_image_3.py` line ~2684: `image_base_size` read from `vae_reso_group.base_size`
- `modeling_hunyuan_image_3.py` line ~1431: `img_ratio` task embeds `<img_size_N>` token
- `image_processor.py`: `build_gen_image_info()` calls `get_target_size()` for bucket snapping
- `tokenization_hunyuan_image_3.py`: `ResolutionGroup`, `size_token_id()`, ratio tokens
