# Eric_Hunyuan3 Quantization Tools

Tools for quantizing HunyuanImage-3.0 models to NF4 format for efficient GPU usage.

## Quick Start

```bash
python hunyuan_quantize_nf4.py \
  --model-path "/path/to/HunyuanImage-3" \
  --output-path "/path/to/HunyuanImage-3-NF4"
```

## What Gets Quantized

**NF4 Quantization Applied To:**
- Feed-forward networks (FFN/MLP layers)
- Large linear transformations
- Expert layers in MoE architecture

**Kept in Full Precision (BF16):**
- VAE encoder/decoder (critical for image quality)
- Attention projection layers (q_proj, k_proj, v_proj, o_proj)
- Patch embedding layers
- Time embedding layers
- Final output layers

## Results

- **Model Size**: 80GB → ~20GB on disk
- **VRAM Usage**: 80GB → ~45GB during inference
- **Quality**: Minimal degradation (attention kept in full precision)
- **Speed**: Slightly slower than full BF16, much faster than CPU offload

## Requirements

- 80GB+ disk space for source model
- 24GB+ VRAM for quantization process
- bitsandbytes library
- transformers library

## Advanced Options

```bash
python hunyuan_quantize_nf4.py \
  --model-path "./HunyuanImage-3" \
  --output-path "./HunyuanImage-3-NF4" \
  --no-double-quant \            # Disable nested quantization
  --compute-dtype float16 \       # Use float16 instead of bfloat16
  --device-map cuda:0             # Specific GPU
```

## Troubleshooting

**Out of memory during quantization:**
- Close other applications
- Use `--device-map auto` (may use CPU)
- Ensure you have swap space available

**Quantized model produces artifacts:**
- Re-run quantization (ensure latest script)
- Check that attention layers are excluded
- Verify compute-dtype is bfloat16

## Files Generated

After quantization, you'll have:
```
HunyuanImage-3-NF4/
├── model-00001-of-00010.safetensors
├── model-00002-of-00010.safetensors
├── ...
├── config.json
├── quantization_metadata.json
├── load_quantized.py (helper script)
└── tokenizer files
```

## Loading Quantized Model

In ComfyUI, use the "Hunyuan 3 Loader (NF4)" node and select "HunyuanImage-3-NF4" from the dropdown.

The quantization config is automatically applied from metadata.
