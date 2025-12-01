---
title: NF4 – Shack Standoff
loader: Hunyuan 3 Loader (NF4)
generator: Hunyuan 3 Generate (Low VRAM)
resolution: 1600x1600
steps: 51
cfg: 7.5 (recommend 6 for gentler highlights)
sampler: FlowMatch Euler
seed: 241553418002667
prompt_rewrite: disabled
---

This render showcases the NF4 loader + Low VRAM generator pairing on the quantized checkpoint. Even with the compressed weights the model follows intricate documentary prompts extremely well—the only tweak I would make next pass is lowering CFG from 7.5 to ~6 to relax the “burnt-in” highlights.

## Settings Snapshot

| Parameter | Value |
| --- | --- |
| Loader | NF4 Low VRAM+ (18 GB budget) |
| Generator | Low VRAM (default 51 steps, FlowMatch Euler) |
| Resolution | 1600 × 1600 |
| Seed | `241553418002667` |
| Guidance Scale | 7.5 (6 recommended) |
| Prompt Rewrite | Disabled |

## Prompt

```text
Description: A photographic scene captures a tense moment at a dilapidated wooden shack situated near a stagnant, algae-covered pond. In the foreground, a young woman with long, dark, unkempt hair sits on a weathered porch swing. She wears a loose-fitting, white ribbed tank top with wide armholes, and frayed, torn denim shorts that are significantly abbreviated in length. A scruffy, medium-sized dog sleeps on the wooden porch planks near her feet. The yard is densely cluttered with debris, including a rusted washing machine, scattered car engine parts, stacked truck tires, and numerous empty glass bottles and crushed aluminum cans. Inside the shack, visible through a grimy, cracked window, a man with a completely bald head and a thick, untrimmed beard leans out. His face is contorted in a pronounced scowl as he points a double-barreled shotgun through the window frame. A young man, holding a bottle of Jack Daniel's whiskey, has just stepped into the yard, his arrival interrupting the scene. In the background, the murky pond water is disturbed by the head and snout of an alligator floating near the shore. The late afternoon sun casts long, deep shadows from the clutter, creating a high-contrast, documentary-style photograph. The aesthetic is raw and unfiltered, depicting a moment of high tension with stark realism. The composition emphasizes the gritty, documentary style through its authentic details and dramatic lighting. This is a stark, realistic photograph capturing a raw, unfiltered moment.
```

