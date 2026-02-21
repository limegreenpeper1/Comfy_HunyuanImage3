"""
Upload all v2 quantized HunyuanImage-3 models to Hugging Face.

Uploads 6 models:
  1. HunyuanImage-3-INT8-v2           (base model, INT8)
  2. HunyuanImage-3-NF4-v2            (base model, NF4)
  3. HunyuanImage-3.0-Instruct-INT8-v2
  4. HunyuanImage-3.0-Instruct-NF4-v2
  5. HunyuanImage-3.0-Instruct-Distil-INT8-v2
  6. HunyuanImage-3.0-Instruct-Distil-NF4-v2

Usage:
    python upload_v2_to_hf.py                         # Upload all 6
    python upload_v2_to_hf.py --model base-nf4        # Upload one
    python upload_v2_to_hf.py --dry-run               # Preview model cards
    python upload_v2_to_hf.py --model base-int8 --dry-run

Expects HF_TOKEN in system environment variables (or will prompt for login).
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from huggingface_hub import HfApi, login, create_repo
except ImportError:
    print("huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


# -- Configuration -----------------------------------------------------------

DEFAULT_USERNAME = "EricRollei"
MODEL_BASE_DIR = Path(r"H:\Testing")

MODELS = {
    # ---- Base (text-to-image only) ----
    "base-int8": {
        "folder": "HunyuanImage-3-INT8-v2",
        "repo_name": "HunyuanImage-3-INT8-v2",
        "display_name": "Hunyuan Image 3.0 Base -- INT8 Quantized (v2)",
        "quant_type": "INT8",
        "quant_method": "INT8 per-channel quantization via bitsandbytes",
        "model_variant": "Base (text-to-image)",
        "base_model": "tencent/HunyuanImage-3.0",
        "steps": 45,
        "cfg_distilled": True,
        "is_instruct": False,
        "disk_size": "~82 GB",
        "vram_usage": "~80 GB weights + ~10-15 GB inference",
        "total_vram": "~90-95 GB",
        "description": (
            "INT8 quantization of the HunyuanImage-3.0 base model (v2). "
            "High-quality text-to-image generation with the Hunyuan 3.0 "
            "Diffusion Transformer + Mixture-of-Experts architecture. "
            "CFG-distilled for single-pass inference."
        ),
        "tags": ["int8"],
        "v2_changes": (
            "v2 uses improved quantization with more precise skip-module "
            "selection, keeping attention projections and critical embedding "
            "layers in full BF16 precision for better image quality."
        ),
    },
    "base-nf4": {
        "folder": "HunyuanImage-3-NF4-v2",
        "repo_name": "HunyuanImage-3-NF4-v2",
        "display_name": "Hunyuan Image 3.0 Base -- NF4 Quantized (v2)",
        "quant_type": "NF4",
        "quant_method": "4-bit NormalFloat (NF4) quantization via bitsandbytes with double quantization",
        "model_variant": "Base (text-to-image)",
        "base_model": "tencent/HunyuanImage-3.0",
        "steps": 45,
        "cfg_distilled": True,
        "is_instruct": False,
        "disk_size": "~47 GB",
        "vram_usage": "~29 GB weights + ~10-15 GB inference",
        "total_vram": "~39-44 GB",
        "description": (
            "NF4 (4-bit) quantization of the HunyuanImage-3.0 base model (v2). "
            "Fits on a single 48GB GPU. High-quality text-to-image generation "
            "with the Hunyuan 3.0 MoE architecture. CFG-distilled for "
            "single-pass inference."
        ),
        "tags": ["nf4", "4bit"],
        "v2_changes": (
            "v2 uses improved quantization with more precise skip-module "
            "selection, keeping attention projections and critical embedding "
            "layers in full BF16 precision for better image quality."
        ),
    },

    # ---- Instruct (full, 50-step) ----
    "instruct-int8": {
        "folder": "HunyuanImage-3.0-Instruct-INT8-v2",
        "repo_name": "HunyuanImage-3.0-Instruct-INT8-v2",
        "display_name": "Hunyuan Image 3.0 Instruct -- INT8 Quantized (v2)",
        "quant_type": "INT8",
        "quant_method": "INT8 per-channel quantization via bitsandbytes",
        "model_variant": "Instruct (Full)",
        "base_model": "tencent/HunyuanImage-3.0-Instruct",
        "steps": 50,
        "cfg_distilled": False,
        "is_instruct": True,
        "disk_size": "~83 GB",
        "vram_usage": "~80 GB weights + ~12-20 GB inference",
        "total_vram": "~92-100 GB",
        "description": (
            "INT8 quantization of the HunyuanImage-3.0 Instruct model (v2). "
            "Supports text-to-image, image editing, multi-image fusion, "
            "and Chain-of-Thought prompt enhancement (recaption/think_recaption)."
        ),
        "tags": ["int8"],
        "v2_changes": (
            "v2 uses improved quantization with more precise skip-module "
            "selection, keeping attention projections and critical embedding "
            "layers in full BF16 precision for better image quality."
        ),
    },
    "instruct-nf4": {
        "folder": "HunyuanImage-3.0-Instruct-NF4-v2",
        "repo_name": "HunyuanImage-3.0-Instruct-NF4-v2",
        "display_name": "Hunyuan Image 3.0 Instruct -- NF4 Quantized (v2)",
        "quant_type": "NF4",
        "quant_method": "4-bit NormalFloat (NF4) quantization via bitsandbytes with double quantization",
        "model_variant": "Instruct (Full)",
        "base_model": "tencent/HunyuanImage-3.0-Instruct",
        "steps": 50,
        "cfg_distilled": False,
        "is_instruct": True,
        "disk_size": "~48 GB",
        "vram_usage": "~29 GB weights + ~12-20 GB inference",
        "total_vram": "~41-49 GB",
        "description": (
            "NF4 (4-bit) quantization of the HunyuanImage-3.0 Instruct model (v2). "
            "Fits on a single 48GB GPU. Supports text-to-image, image editing, "
            "multi-image fusion, and Chain-of-Thought prompt enhancement."
        ),
        "tags": ["nf4", "4bit"],
        "v2_changes": (
            "v2 uses improved quantization with more precise skip-module "
            "selection, keeping attention projections and critical embedding "
            "layers in full BF16 precision for better image quality."
        ),
    },

    # ---- Instruct Distil (CFG-distilled, 8-step) ----
    "instruct-distil-int8": {
        "folder": "HunyuanImage-3.0-Instruct-Distil-INT8-v2",
        "repo_name": "HunyuanImage-3.0-Instruct-Distil-INT8-v2",
        "display_name": "Hunyuan Image 3.0 Instruct Distil -- INT8 Quantized (v2)",
        "quant_type": "INT8",
        "quant_method": "INT8 per-channel quantization via bitsandbytes",
        "model_variant": "Instruct Distil (CFG-Distilled, 8-step)",
        "base_model": "tencent/HunyuanImage-3.0-Instruct-Distil",
        "steps": 8,
        "cfg_distilled": True,
        "is_instruct": True,
        "disk_size": "~83 GB",
        "vram_usage": "~80 GB weights + ~12-20 GB inference",
        "total_vram": "~92-100 GB",
        "description": (
            "INT8 quantization of the HunyuanImage-3.0 Instruct Distil model (v2). "
            "CFG-distilled for ~6x faster generation (8 steps vs 50). "
            "Same quality as the full Instruct model with dramatically faster inference."
        ),
        "tags": ["int8", "distilled"],
        "v2_changes": (
            "v2 uses improved quantization with more precise skip-module "
            "selection, keeping attention projections and critical embedding "
            "layers in full BF16 precision for better image quality."
        ),
    },
    "instruct-distil-nf4": {
        "folder": "HunyuanImage-3.0-Instruct-Distil-NF4-v2",
        "repo_name": "HunyuanImage-3.0-Instruct-Distil-NF4-v2",
        "display_name": "Hunyuan Image 3.0 Instruct Distil -- NF4 Quantized (v2)",
        "quant_type": "NF4",
        "quant_method": "4-bit NormalFloat (NF4) quantization via bitsandbytes with double quantization",
        "model_variant": "Instruct Distil (CFG-Distilled, 8-step)",
        "base_model": "tencent/HunyuanImage-3.0-Instruct-Distil",
        "steps": 8,
        "cfg_distilled": True,
        "is_instruct": True,
        "disk_size": "~48 GB",
        "vram_usage": "~29 GB weights + ~12-20 GB inference",
        "total_vram": "~41-49 GB",
        "description": (
            "NF4 (4-bit) quantization of the HunyuanImage-3.0 Instruct Distil model (v2). "
            "The most accessible option -- fits on a single 48GB GPU with ~6x faster "
            "generation (8 steps vs 50). Best balance of speed, quality, and VRAM."
        ),
        "tags": ["nf4", "4bit", "distilled"],
        "v2_changes": (
            "v2 uses improved quantization with more precise skip-module "
            "selection, keeping attention projections and critical embedding "
            "layers in full BF16 precision for better image quality."
        ),
    },
}


# -- Model Card Templates ----------------------------------------------------

def _make_base_model_card(info: dict) -> str:
    """Generate HF model card for a base (text-to-image) model."""

    if info["quant_type"] == "NF4":
        hw_note = (
            "- **Single 48GB GPU** (RTX 6000 Ada, RTX PRO 5000, A6000)\n"
            "- With block swap: may work on 24GB GPUs (swapping ~20 blocks)\n"
        )
    else:
        hw_note = (
            "- **NVIDIA RTX 6000 Blackwell (96GB)** -- fits entirely with headroom\n"
            "- With block swap (4-8 blocks): fits on 64-80GB GPUs\n"
            "- **NVIDIA RTX 6000 Ada (48GB)** -- requires significant block swap\n"
        )

    base_tags = [
        "Hunyuan", "hunyuan", "quantization", info["quant_type"].lower(),
        "comfyui", "custom-nodes", "autoregressive", "DiT",
        "HunyuanImage-3.0", "text-to-image", "bitsandbytes",
    ]
    all_tags = list(dict.fromkeys(base_tags + info.get("tags", [])))
    tags_yaml = "\n".join(f"- {t}" for t in all_tags)

    vram_parts = info["vram_usage"].split("+")
    vram_weight = vram_parts[0].strip()
    vram_infer = vram_parts[1].strip() if len(vram_parts) > 1 else "~10-15 GB"

    card = f"""---
license: other
license_name: tencent-hunyuan-community
license_link: https://huggingface.co/tencent/HunyuanImage-3.0/blob/main/LICENSE.txt
base_model: {info['base_model']}
pipeline_tag: text-to-image
library_name: transformers
tags:
{tags_yaml}
---

# {info['display_name']}

{info['description']}

## What's New in v2

{info['v2_changes']}

## Key Features

- **Text-to-image generation** with the Hunyuan 3.0 MoE architecture
- **{info['quant_type']} quantized** -- {info['disk_size']} on disk
- **{info['steps']} diffusion steps** (CFG-distilled, single-pass)
- **Block swap support** -- offload transformer blocks to CPU for lower VRAM
- **ComfyUI ready** -- works with [Comfy_HunyuanImage3](https://github.com/EricRollei/Comfy_HunyuanImage3) nodes

## VRAM Requirements

| Component | Memory |
|-----------|--------|
| Weight Loading | {vram_weight} |
| Inference (additional) | {vram_infer} |
| **Total** | **{info['total_vram']}** |

**Recommended Hardware:**

{hw_note}

## Model Details

- **Architecture:** HunyuanImage-3.0 Mixture-of-Experts Diffusion Transformer
- **Parameters:** 80B total, 13B active per token (top-K MoE routing)
- **Variant:** {info['model_variant']}
- **Quantization:** {info['quant_method']}
- **Diffusion Steps:** {info['steps']}
- **Default Guidance Scale:** 7.0
- **Resolution:** Up to 2048x2048
- **Language:** English and Chinese prompts

## Quantization Details

**Layers quantized to {info['quant_type']}:**
- Feed-forward networks (FFN/MLP layers)
- Expert layers in MoE architecture (64 experts per layer)
- Large linear transformations

**Kept in full precision (BF16):**
- VAE encoder/decoder (critical for image quality)
- Attention projection layers (q_proj, k_proj, v_proj, o_proj)
- Patch embedding layers
- Time embedding layers
- Vision model (SigLIP2)
- Final output layers

## Usage

### ComfyUI (Recommended)

This model is designed to work with the [Comfy_HunyuanImage3](https://github.com/EricRollei/Comfy_HunyuanImage3) custom nodes:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/EricRollei/Comfy_HunyuanImage3
```

1. Download this model to your preferred models directory
2. Use the **"Hunyuan 3 V2 Unified"** node
3. Point the model path to this folder and select `{info['quant_type'].lower()}` precision
4. Set `blocks_to_swap` to -1 (auto) or a manual value based on your VRAM

## Block Swap

Block swap allows running INT8 and BF16 models on GPUs with less VRAM than the
full model requires. The system keeps N transformer blocks on CPU and swaps them
to GPU on demand during each diffusion step.

| blocks_to_swap | VRAM Saved | Recommended For |
|---------------|------------|-----------------|
| 0 | 0 GB | 96GB+ GPU (no swap needed) |
| 4 | ~{4 * (2.5 if info['quant_type'] == 'INT8' else 1.2):.0f} GB | 80-90GB GPU |
| 8 | ~{8 * (2.5 if info['quant_type'] == 'INT8' else 1.2):.0f} GB | 64-80GB GPU |
| 16 | ~{16 * (2.5 if info['quant_type'] == 'INT8' else 1.2):.0f} GB | 48-64GB GPU |
| -1 (auto) | varies | Let the system decide |

## Original Model

This is a quantized derivative of [Tencent's HunyuanImage-3.0]({info['base_model']}).

- **License:** [Tencent Hunyuan Community License](https://huggingface.co/tencent/HunyuanImage-3.0/blob/main/LICENSE.txt)

## Credits

- **Original Model:** [Tencent Hunyuan Team](https://huggingface.co/tencent)
- **Quantization:** Eric Rollei
- **ComfyUI Integration:** [Comfy_HunyuanImage3](https://github.com/EricRollei/Comfy_HunyuanImage3)

## License

This model inherits the license from the original Hunyuan Image 3.0 model:
[Tencent Hunyuan Community License](https://huggingface.co/tencent/HunyuanImage-3.0/blob/main/LICENSE.txt)
"""
    return card


def _make_instruct_model_card(info: dict) -> str:
    """Generate HF model card for an Instruct model."""

    distil_section = ""
    if info["cfg_distilled"]:
        distil_section = (
            "\n### Distillation\n\n"
            "This is the **CFG-Distilled** variant:\n"
            "- Only **8 diffusion steps** needed (vs 50 for the full Instruct model)\n"
            "- **~6x faster** image generation\n"
            "- No quality loss -- distilled to match the full model's output\n"
            "- `cfg_distilled: true` means no classifier-free guidance needed\n"
        )

    if info["quant_type"] == "NF4":
        hw_note = (
            "- **Single 48GB GPU** (RTX 6000 Ada, RTX PRO 5000, A6000)\n"
            "- With block swap: may work on 24GB GPUs (swapping ~20 blocks)\n"
        )
    else:
        hw_note = (
            "- **NVIDIA RTX 6000 Blackwell (96GB)** -- fits entirely with headroom\n"
            "- With block swap (4-8 blocks): fits on 64-80GB GPUs\n"
            "- **NVIDIA RTX 6000 Ada (48GB)** -- requires significant block swap\n"
        )

    base_tags = [
        "Hunyuan", "hunyuan", "quantization", info["quant_type"].lower(),
        "comfyui", "custom-nodes", "autoregressive", "DiT",
        "HunyuanImage-3.0", "instruct", "image-editing", "bitsandbytes",
    ]
    all_tags = list(dict.fromkeys(base_tags + info.get("tags", [])))
    tags_yaml = "\n".join(f"- {t}" for t in all_tags)

    vram_parts = info["vram_usage"].split("+")
    vram_weight = vram_parts[0].strip()
    vram_infer = vram_parts[1].strip() if len(vram_parts) > 1 else "~12-20 GB"
    speed_note = "(CFG-distilled)" if info["cfg_distilled"] else "(full quality)"

    card = f"""---
license: other
license_name: tencent-hunyuan-community
license_link: https://huggingface.co/tencent/HunyuanImage-3.0/blob/main/LICENSE.txt
base_model: {info['base_model']}
pipeline_tag: text-to-image
library_name: transformers
tags:
{tags_yaml}
---

# {info['display_name']}

{info['description']}

## What's New in v2

{info['v2_changes']}

## Key Features

- **Instruct model** -- supports text-to-image, image editing, multi-image fusion
- **Chain-of-Thought** -- built-in `think_recaption` mode for highest quality
- **{info['quant_type']} quantized** -- {info['disk_size']} on disk
- **{info['steps']} diffusion steps** {speed_note}
- **Block swap support** -- offload transformer blocks to CPU for lower VRAM
- **ComfyUI ready** -- works with [Comfy_HunyuanImage3](https://github.com/EricRollei/Comfy_HunyuanImage3) nodes

## VRAM Requirements

| Component | Memory |
|-----------|--------|
| Weight Loading | {vram_weight} |
| Inference (additional) | {vram_infer} |
| **Total** | **{info['total_vram']}** |

**Recommended Hardware:**

{hw_note}

## Model Details

- **Architecture:** HunyuanImage-3.0 Mixture-of-Experts Diffusion Transformer
- **Parameters:** 80B total, 13B active per token (top-K MoE routing)
- **Variant:** {info['model_variant']}
- **Quantization:** {info['quant_method']}
- **Diffusion Steps:** {info['steps']}
- **Default Guidance Scale:** 2.5
- **Resolution:** Up to 2048x2048
- **Language:** English and Chinese prompts
{distil_section}
## Quantization Details

**Layers quantized to {info['quant_type']}:**
- Feed-forward networks (FFN/MLP layers)
- Expert layers in MoE architecture (64 experts per layer)
- Large linear transformations

**Kept in full precision (BF16):**
- VAE encoder/decoder (critical for image quality)
- Attention projection layers (q_proj, k_proj, v_proj, o_proj)
- Patch embedding layers
- Time embedding layers
- Vision model (SigLIP2)
- Final output layers

## Usage

### ComfyUI (Recommended)

This model is designed to work with the [Comfy_HunyuanImage3](https://github.com/EricRollei/Comfy_HunyuanImage3) custom nodes:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/EricRollei/Comfy_HunyuanImage3
```

1. Download this model to your preferred models directory
2. Use the **"Hunyuan 3 Instruct Loader"** node
3. Select this model folder and choose `{info['quant_type'].lower()}` precision
4. Connect to the **"Hunyuan 3 Instruct Generate"** node for text-to-image
5. Or use **"Hunyuan 3 Instruct Edit"** for image editing
6. Or use **"Hunyuan 3 Instruct Multi-Fusion"** for combining multiple images

### Bot Task Modes

The Instruct model supports three generation modes:

| Mode | Description | Speed |
|------|-------------|-------|
| `image` | Direct text-to-image, prompt used as-is | Fastest |
| `recaption` | Model rewrites prompt into detailed description, then generates | Medium |
| `think_recaption` | CoT reasoning -> prompt enhancement -> generation (best quality) | Slowest |

## Block Swap

Block swap allows running INT8 and BF16 models on GPUs with less VRAM than the
full model requires. The system keeps N transformer blocks on CPU and swaps them
to GPU on demand during each diffusion step.

| blocks_to_swap | VRAM Saved | Recommended For |
|---------------|------------|-----------------|
| 0 | 0 GB | 96GB+ GPU (no swap needed) |
| 4 | ~{4 * (2.5 if info['quant_type'] == 'INT8' else 1.2):.0f} GB | 80-90GB GPU |
| 8 | ~{8 * (2.5 if info['quant_type'] == 'INT8' else 1.2):.0f} GB | 64-80GB GPU |
| 16 | ~{16 * (2.5 if info['quant_type'] == 'INT8' else 1.2):.0f} GB | 48-64GB GPU |
| -1 (auto) | varies | Let the system decide |

## Original Model

This is a quantized derivative of [Tencent's HunyuanImage-3.0 Instruct]({info['base_model']}).

- **License:** [Tencent Hunyuan Community License](https://huggingface.co/tencent/HunyuanImage-3.0/blob/main/LICENSE.txt)

## Credits

- **Original Model:** [Tencent Hunyuan Team](https://huggingface.co/tencent)
- **Quantization:** Eric Rollei
- **ComfyUI Integration:** [Comfy_HunyuanImage3](https://github.com/EricRollei/Comfy_HunyuanImage3)

## License

This model inherits the license from the original Hunyuan Image 3.0 model:
[Tencent Hunyuan Community License](https://huggingface.co/tencent/HunyuanImage-3.0/blob/main/LICENSE.txt)
"""
    return card


def make_model_card(info: dict) -> str:
    """Route to appropriate card template based on model type."""
    if info.get("is_instruct", False):
        return _make_instruct_model_card(info)
    else:
        return _make_base_model_card(info)


# -- Upload Logic -------------------------------------------------------------

def upload_model(key: str, info: dict, api: HfApi, username: str, dry_run: bool = False):
    """Upload a single model to Hugging Face."""
    folder_path = MODEL_BASE_DIR / info["folder"]
    repo_id = f"{username}/{info['repo_name']}"

    print(f"\n{'='*60}")
    print(f"  {info['display_name']}")
    print(f"   Folder: {folder_path}")
    print(f"   Repo:   {repo_id}")
    print(f"   Size:   {info['disk_size']}")
    print(f"{'='*60}")

    if not folder_path.exists():
        print(f"   Folder not found: {folder_path}")
        return False

    # Count files
    file_count = sum(1 for _ in folder_path.rglob("*") if _.is_file())
    print(f"   Files:  {file_count}")

    # Generate model card
    card_content = make_model_card(info)

    if dry_run:
        print(f"\n   Model card preview (first 30 lines):")
        for line in card_content.split("\n")[:30]:
            print(f"      {line}")
        print(f"      ... ({len(card_content.splitlines())} total lines)")
        print(f"\n   DRY RUN -- no upload performed")
        return True

    # Write README.md into model folder
    readme_path = folder_path / "README.md"
    print(f"   Writing model card...")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card_content)

    # Create repo (idempotent)
    print(f"   Creating repository '{repo_id}'...")
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"   Repository ready")
    except Exception as e:
        print(f"   Error creating repo: {e}")
        return False

    # Upload folder
    print(f"   Uploading files ({info['disk_size']})...")
    print(f"   This will take a while depending on your internet speed.")
    try:
        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=[
                ".git", ".DS_Store", "__pycache__", "*.bak",
                "*.pyc", ".gitattributes",
            ],
        )
        print(f"   Upload complete!")
        print(f"   https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"   Upload failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload v2 quantized HunyuanImage-3 models to Hugging Face"
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Which model to upload (default: all)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview model cards without uploading"
    )
    parser.add_argument(
        "--username", "-u",
        default=None,
        help=f"HF username (default: auto-detect, fallback: {DEFAULT_USERNAME})"
    )
    args = parser.parse_args()

    print("Hugging Face Model Uploader -- HunyuanImage-3 v2 Quantizations")
    print("=" * 65)

    # Authenticate
    if not args.dry_run:
        print("\nAuthenticating...")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            print("   Found HF_TOKEN in environment")
            login(token=token)
        else:
            print("   No HF_TOKEN found -- attempting interactive login")
            login()

    api = HfApi()

    # Determine username
    username = args.username
    if not username and not args.dry_run:
        try:
            username = api.whoami()["name"]
            print(f"   Logged in as: {username}")
        except Exception:
            username = DEFAULT_USERNAME
            print(f"   Could not detect username, using: {username}")
    elif not username:
        username = DEFAULT_USERNAME

    # Select models
    if args.model == "all":
        to_upload = MODELS
    else:
        to_upload = {args.model: MODELS[args.model]}

    action = "preview" if args.dry_run else "upload"
    print(f"\nModels to {action}: {len(to_upload)}")
    for key, info in to_upload.items():
        print(f"   - {info['display_name']} ({info['disk_size']})")

    if not args.dry_run:
        total_gb = 0
        for info in to_upload.values():
            # Parse approximate size from disk_size string
            size_str = info["disk_size"].replace("~", "").replace("GB", "").strip()
            try:
                total_gb += float(size_str)
            except ValueError:
                total_gb += 50  # fallback estimate
        print(f"\n   Total upload size: ~{total_gb:.0f} GB")
        confirm = input("\n   Proceed? (y/n): ").strip().lower()
        if confirm != "y":
            print("   Cancelled.")
            return

    # Upload each model
    results = {}
    for key, info in to_upload.items():
        results[key] = upload_model(key, info, api, username, args.dry_run)

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    for key, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"   [{status}] {MODELS[key]['display_name']}")

    if not args.dry_run and all(results.values()):
        print(f"\nAll uploads complete! View at:")
        for key in results:
            info = MODELS[key]
            print(f"   https://huggingface.co/{username}/{info['repo_name']}")


if __name__ == "__main__":
    main()
