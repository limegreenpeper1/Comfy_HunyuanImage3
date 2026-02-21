# Latent Control Nodes — Rollback Guide

**Feature added:** February 17, 2026  
**Files changed:** 3 (1 new, 2 modified)

---

## Quick Rollback (recommended)

Delete the new node file and remove its registration. The 6-line passthrough in shared.py is harmless (never triggered without the latent nodes) and can stay.

1. **Delete** `hunyuan_latent_nodes.py`
2. **Remove** this block from `__init__.py`:
   ```python
   try:
       from .hunyuan_latent_nodes import NODE_CLASS_MAPPINGS as LATENT_MAPPINGS
       from .hunyuan_latent_nodes import NODE_DISPLAY_NAME_MAPPINGS as LATENT_DISPLAY
       NODE_CLASS_MAPPINGS.update(LATENT_MAPPINGS)
       NODE_DISPLAY_NAME_MAPPINGS.update(LATENT_DISPLAY)
   except Exception as e:
       print(f"[Eric_Hunyuan3] Latent control nodes not available: {e}")
   ```
3. Restart ComfyUI.

---

## Full Rollback (revert all changes)

If you also want to remove the shared.py passthrough (not necessary, but for completeness):

### In `hunyuan_shared.py` — `new_generate_image()` function

Remove these 3 lines (search for `# LATENT CONTROL`):

```python
latents = kwargs.pop("latents", None)  # LATENT CONTROL: extract custom latents
```

```python
if latents is not None:  # LATENT CONTROL: pass through custom latents
    gen_kwargs["latents"] = latents
```

### In `hunyuan_shared.py` — `new_pipeline_call()` function

Remove these 3 lines (search for `# LATENT CONTROL`):

```python
# LATENT CONTROL: extract latents from model_kwargs → top-level pipeline kwarg
if 'latents' in model_kwargs:
    lat = model_kwargs.pop('latents')
    if kwargs.get('latents') is None:
        kwargs['latents'] = lat
```

### Summary of all lines to revert

| File | What to do |
|------|-----------|
| `hunyuan_latent_nodes.py` | Delete entire file |
| `__init__.py` | Remove the try/except import block (5 lines) |
| `hunyuan_shared.py` | Remove 6 lines marked with `# LATENT CONTROL` comments |

After reverting, all existing nodes work exactly as they did before this feature was added.
