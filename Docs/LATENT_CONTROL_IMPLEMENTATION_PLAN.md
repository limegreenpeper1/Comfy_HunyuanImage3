# Latent Control Nodes — Implementation Plan

**Date:** February 16, 2026  
**Status:** Implemented  
**Risk Level:** Low (fully reversible, no changes to generation behavior when not used)

---

## Goal

Add optional latent/noise control to HunyuanImage-3.0 generation, enabling:
- **img2img**: Start from a reference image instead of pure noise
- **Custom noise patterns**: Frequency-biased, spatially structured noise
- **Seed-based latent reuse**: Generate a latent once, reuse across prompts for consistent composition

All new functionality lives in a **separate node file** that can be deleted to fully revert. The only change to existing code is a small, backward-compatible passthrough in `hunyuan_shared.py`.

---

## Architecture Overview

### Call Chain (current)
```
V2 Node → model.generate_image(**gen_kwargs)
  → [our patched generate_image] → model._generate(**model_inputs, **gen_kwargs)
    → [upstream _generate] → self.pipeline(model_kwargs=kwargs)
      → [our patched pipeline.__call__] → original pipeline.__call__(**kwargs)
        → pipeline.prepare_latents(latents=None) → randn_tensor(...)
```

### Call Chain (proposed)
```
Latent Control Node → model.generate_image(**gen_kwargs, latents=custom_latent)
  → [our patched generate_image] → extracts latents, passes to _generate
    → [upstream _generate] → self.pipeline(latents=latents, model_kwargs=kwargs)
                                           ^^^^^^^^^^^^^^^^
                                           NEW: explicit parameter
      → pipeline.prepare_latents(latents=custom_latent) → uses provided latent
```

### Key Insight
The upstream `pipeline.__call__()` **already accepts** `latents=` as a named parameter (line 698 of `hunyuan_image_3_pipeline.py`), and `prepare_latents()` **already handles** pre-provided latents (line 649). We just need to get the latent from our node into that parameter.

---

## Changes Required

### 1. `hunyuan_shared.py` — Patch modification (MINIMAL)

**What changes:** The `patch_hunyuan_generate_image()` function — specifically two locations:

#### 1a. `new_generate_image()` — extract `latents` from kwargs

```python
# CURRENT (line ~2335):
callback_on_step_end = kwargs.pop("callback_on_step_end", None)

# PROPOSED:
callback_on_step_end = kwargs.pop("callback_on_step_end", None)
latents = kwargs.pop("latents", None)  # NEW: extract custom latents
```

Then pass `latents` into the final `gen_kwargs`:

```python
# CURRENT (line ~2381):
gen_kwargs = kwargs.copy()
if callback_on_step_end is not None:
    gen_kwargs["callback_on_step_end"] = callback_on_step_end

# PROPOSED:
gen_kwargs = kwargs.copy()
if callback_on_step_end is not None:
    gen_kwargs["callback_on_step_end"] = callback_on_step_end
if latents is not None:
    gen_kwargs["latents"] = latents  # NEW: pass through custom latents
```

#### 1b. `new_pipeline_call()` — extract `latents` from model_kwargs

The `_generate()` method puts everything into `model_kwargs` when calling the pipeline. We need to extract `latents` from there and pass it as a top-level kwarg:

```python
# CURRENT (line ~2408):
def new_pipeline_call(self, *args, **kwargs):
    model_kwargs = kwargs.get('model_kwargs')
    if model_kwargs and isinstance(model_kwargs, dict):
        if 'callback_on_step_end' in model_kwargs:
            cb = model_kwargs.pop('callback_on_step_end')
            if kwargs.get('callback_on_step_end') is None:
                kwargs['callback_on_step_end'] = cb
    return original_call(self, *args, **kwargs)

# PROPOSED:
def new_pipeline_call(self, *args, **kwargs):
    model_kwargs = kwargs.get('model_kwargs')
    if model_kwargs and isinstance(model_kwargs, dict):
        if 'callback_on_step_end' in model_kwargs:
            cb = model_kwargs.pop('callback_on_step_end')
            if kwargs.get('callback_on_step_end') is None:
                kwargs['callback_on_step_end'] = cb
        # NEW: extract latents from model_kwargs → top-level pipeline kwarg
        if 'latents' in model_kwargs:
            lat = model_kwargs.pop('latents')
            if kwargs.get('latents') is None:
                kwargs['latents'] = lat
    return original_call(self, *args, **kwargs)
```

**Risk assessment:** When `latents` is not in kwargs (i.e., ALL existing nodes), these lines are never reached. The `kwargs.pop("latents", None)` returns `None`, and the `if latents is not None` check skips the passthrough. Behavior is 100% identical to current code.

### 2. `hunyuan_latent_nodes.py` — NEW FILE (all new nodes)

This file contains all new ComfyUI nodes. Deleting it fully reverts all new functionality.

#### Node: `HunyuanEmptyLatent`

Creates a noise latent for a given resolution and seed.

**Inputs:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| resolution | COMBO | 1024×1024 | Same presets as V2 unified |
| seed | INT | 0 | Random seed for noise generation |
| batch_size | INT | 1 | Batch size |

**Output:** `HUNYUAN_LATENT` (dict with `{"latent": tensor, "shape": tuple, "seed": int}`)

**Implementation:**
```python
def generate(self, resolution, seed, batch_size):
    h, w = parse_resolution(resolution)
    # Match pipeline's latent shape computation exactly
    vae_downsample = (8, 8)  # from model config
    latent_channels = 16     # from model config
    shape = (batch_size, latent_channels, h // vae_downsample[0], w // vae_downsample[1])
    
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn(shape, generator=generator, dtype=torch.bfloat16)
    
    return ({"latent": latent, "shape": shape, "seed": seed},)
```

#### Node: `HunyuanImageToLatent`

VAE-encodes a reference image into latent space, optionally adding noise for img2img.

**Inputs:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| image | IMAGE | required | Reference image (ComfyUI IMAGE tensor) |
| denoise_strength | FLOAT | 0.75 | 0.0 = exact reproduction, 1.0 = pure noise (ignore reference) |
| seed | INT | 0 | Seed for the noise component |
| model_name | COMBO | required | Model to use for VAE encoding |

**Output:** `HUNYUAN_LATENT`

**Implementation:**
- Uses `ModelCacheV2.get()` to access the already-loaded model (no separate load)
- Calls `model.vae_encode()` on the reference image
- Applies noise mixing: `noisy_latent = (1 - strength) * clean_latent + strength * noise`
  - This matches the flow matching formula — at `strength=0.75`, the image starts 75% noisy
- The scheduler `init_noise_sigma` scaling will be applied by `pipeline.prepare_latents()`

**Important note on `init_noise_sigma`:** The pipeline's `prepare_latents()` applies `latents = latents * scheduler.init_noise_sigma` when the scheduler has that attribute. For pre-provided latents, we need to account for this. Two options:
  - Option A: Pre-divide our latent by `init_noise_sigma` so the pipeline multiplication gets the right value
  - Option B: The `FlowMatchDiscreteScheduler` in Hunyuan doesn't set `init_noise_sigma` (it's inherited from `SchedulerMixin` with default 1.0), so the multiplication is a no-op. **Need to verify this at runtime.**

#### Node: `HunyuanLatentNoise`

Applies noise transformations to an existing latent.

**Inputs:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| latent | HUNYUAN_LATENT | required | Input latent |
| operation | COMBO | "low_pass_filter" | Noise shaping operation |
| strength | FLOAT | 0.5 | Effect strength |
| seed | INT | 0 | Seed for any random operations |

**Operations:**
- `low_pass_filter`: Gaussian blur the noise → stronger macro composition
- `high_pass_filter`: Remove low frequencies → more detail variation, less composition
- `blend_seeds`: Spherical interpolation (slerp) between current latent and a new seed's noise
- `spatial_mask_left_right`: Different seeds for left/right halves
- `spatial_mask_top_bottom`: Different seeds for top/bottom halves
- `amplify_center`: Boost noise magnitude in center region
- `invert`: Negate the noise pattern

**Output:** `HUNYUAN_LATENT`

#### Node: `HunyuanGenerateWithLatent`

Generation node that accepts an optional custom latent. Reuses the model from `ModelCacheV2`.

**Inputs:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_name | COMBO | required | Model name (must already be loaded) |
| prompt | STRING | required | Text prompt |
| latent | HUNYUAN_LATENT | optional | Custom initial latent. If not connected, behaves like normal generation. |
| seed | INT | -1 | Seed (used normally if no latent provided) |
| num_inference_steps | INT | 30 | Denoising steps |
| guidance_scale | FLOAT | 5.0 | CFG scale |
| flow_shift | FLOAT | 2.8 | Flow matching shift |
| resolution | COMBO | ... | Resolution (ignored if latent provided — uses latent's shape) |
| post_action | COMBO | "keep_loaded" | Model lifecycle |

**Output:** `IMAGE`, `STRING` (prompt used)

**Implementation:**
```python
def generate(self, model_name, prompt, seed, ..., latent=None):
    cached = ModelCacheV2().get(...)
    model = cached.model
    
    gen_kwargs = {
        "prompt": prompt,
        "image_size": image_size,
        "seed": seed,
        "stream": False,
    }
    
    if latent is not None:
        gen_kwargs["latents"] = latent["latent"]
        # Resolution must match latent shape
    
    result = model.generate_image(**gen_kwargs)
    ...
```

### 3. `__init__.py` — Registration (1 line change)

Add in the existing pattern:
```python
try:
    from .hunyuan_latent_nodes import NODE_CLASS_MAPPINGS as LATENT_MAPPINGS
    from .hunyuan_latent_nodes import NODE_DISPLAY_NAME_MAPPINGS as LATENT_DISPLAY
    NODE_CLASS_MAPPINGS.update(LATENT_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(LATENT_DISPLAY)
except Exception as e:
    print(f"[Eric_Hunyuan3] Latent control nodes not available: {e}")
```

---

## Rollback Procedure

### Full Revert (nuclear option)
1. Delete `hunyuan_latent_nodes.py`
2. Remove the `try/except` block from `__init__.py`
3. Revert the 2 small additions in `hunyuan_shared.py` (remove the `latents` extraction lines)
4. Done. All existing nodes work exactly as before.

### Partial Revert (keep passthrough, remove nodes)
1. Delete `hunyuan_latent_nodes.py`
2. Remove the `try/except` block from `__init__.py`
3. The shared.py changes are harmless (never triggered) — can leave in place.

### If shared.py changes cause issues (shouldn't be possible, but just in case)
The git diff would show exactly 6 new lines in `hunyuan_shared.py`. Revert those 6 lines:
- 1 line: `latents = kwargs.pop("latents", None)`
- 2 lines: `if latents is not None:` + `gen_kwargs["latents"] = latents`
- 3 lines: `if 'latents' in model_kwargs:` + `lat = model_kwargs.pop('latents')` + `if kwargs.get('latents') is None: kwargs['latents'] = lat`

---

## Technical Risks & Mitigations

### Risk 1: `init_noise_sigma` scaling
**Issue:** `prepare_latents()` multiplies provided latents by `scheduler.init_noise_sigma`.
**Mitigation:** Verify at runtime. For `FlowMatchDiscreteScheduler` with the Hunyuan config, `init_noise_sigma` should be 1.0 (no-op). We'll add a runtime check and log a warning if it's not 1.0.

### Risk 2: Latent shape mismatch
**Issue:** If user provides wrong-shaped latent, the pipeline will crash.
**Mitigation:** Validate shape in `HunyuanGenerateWithLatent` before calling generate. If `latent` is provided, derive resolution from its shape and ignore the resolution dropdown. Log the resolution being used.

### Risk 3: CFG duplication
**Issue:** The pipeline does `latent_model_input = torch.cat([latents] * cfg_factor)` for classifier-free guidance (doubles the latent). This is handled internally and works fine with provided latents — same as SD.
**Mitigation:** No action needed. This is standard behavior.

### Risk 4: dtype mismatch
**Issue:** Pipeline casts to `torch.bfloat16`. Our latents should match.
**Mitigation:** Generate/convert latents in bfloat16 in the node.

### Risk 5: Device mismatch
**Issue:** `prepare_latents()` does `latents = latents.to(device)` which handles this.
**Mitigation:** No action needed. We can keep latents on CPU and let the pipeline move them.

---

## Constants to Extract from Model

These values come from the model's config and are needed for correct latent shape computation:

```python
# From model.config.vae_downsample_factor — [16, 16] for Hunyuan (not 8!)
# From model.config.vae["latent_channels"] — 32 for Hunyuan (not 16!)
# From model.config.vae["scaling_factor"]  — 0.562679178327931
```

For the `HunyuanEmptyLatent` node that works without a loaded model, we'll hardcode these as defaults with a note. When used with `HunyuanGenerateWithLatent`, the shape is validated against the actual model config.

---

## Testing Plan

### Phase 1: Passthrough verification (shared.py changes only)
1. Apply the 6-line change to `hunyuan_shared.py`
2. Run existing V2 unified node — verify behavior is identical
3. Run existing Instruct nodes — verify behavior is identical
4. Run existing HighRes nodes — verify behavior is identical
5. All should produce identical images for the same seed

### Phase 2: Empty latent → generate
1. Create `HunyuanEmptyLatent` with seed=42, 1024×1024
2. Connect to `HunyuanGenerateWithLatent`
3. Compare output to V2 unified with seed=42, 1024×1024
4. Should produce the same image (validates latent shape is correct)

### Phase 3: img2img
1. Load a reference image
2. `HunyuanImageToLatent` with denoise_strength=0.3 → should be recognizably similar
3. denoise_strength=0.7 → should be loosely similar
4. denoise_strength=1.0 → should be completely different (pure noise)

### Phase 4: Noise control
1. `HunyuanEmptyLatent` → `HunyuanLatentNoise(low_pass_filter)` → generate
2. Compare to unfiltered — should show same broad composition but different fine detail
3. Test spatial blending — different subjects in left/right halves

---

## Implementation Order

1. **shared.py passthrough** (6 lines) → test with existing nodes → confirm no regression
2. **HunyuanEmptyLatent** + **HunyuanGenerateWithLatent** (minimum viable) → test basic latent injection
3. **HunyuanImageToLatent** → test img2img
4. **HunyuanLatentNoise** → test noise shaping (can be added incrementally)

---

## File Summary

| File | Change | Lines | Reversible |
|------|--------|-------|------------|
| `hunyuan_shared.py` | Add latent passthrough in 2 existing patches | +6 lines | Delete 6 lines |
| `hunyuan_latent_nodes.py` | **NEW FILE** — all nodes | ~500 lines est. | Delete file |
| `__init__.py` | Add try/except import | +5 lines | Delete 5 lines |

**Total risk to existing functionality: Near zero.** The shared.py changes are conditional on a kwarg that no existing node provides.
