"""
HunyuanImage-3.0 NF4 Quantization Script
Quantize the full BF16 model to NF4 format for reduced VRAM usage (~45GB)

Author: Eric Hiss (GitHub: EricRollei)
Contact: [eric@historic.camera, eric@rollei.us]
License: Dual License (Non-Commercial and Commercial Use)
Copyright (c) 2025 Eric Hiss. All rights reserved.

Dual License:
1. Non-Commercial Use: This software is licensed under the terms of the
   Creative Commons Attribution-NonCommercial 4.0 International License.
   To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
   
2. Commercial Use: For commercial use, a separate license is required.
   Please contact Eric Hiss at [eric@historic.camera, eric@rollei.us] for licensing options.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
PARTICULAR PURPOSE AND NONINFRINGEMENT.

Note: HunyuanImage-3.0 model is subject to Tencent's Apache 2.0 license.
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HunyuanQuantizer:
    """
    Handles quantization of HunyuanImage-3.0 model to NF4 format.
    
    This class provides a clean interface for quantizing the model and
    saving it in a format optimized for fast loading on GPU.
    """
    
    def __init__(
        self,
        model_path: str,
        output_path: str,
        compute_dtype: torch.dtype = torch.bfloat16,
        use_double_quant: bool = True,
        device_map: str = "auto"
    ):
        """
        Initialize the quantizer.
        
        Args:
            model_path: Path to the source model (or HuggingFace model ID)
            output_path: Path where quantized model will be saved
            compute_dtype: Computation dtype (bfloat16 recommended for quality)
            use_double_quant: Whether to use nested quantization (saves ~0.4 bits/param)
            device_map: Device mapping strategy ("auto", "cuda:0", etc.)
        """
        self.model_path = model_path
        self.output_path = Path(output_path)
        self.compute_dtype = compute_dtype
        self.use_double_quant = use_double_quant
        self.device_map = device_map
        
        # Validate paths
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized quantizer:")
        logger.info(f"  Source: {model_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Compute dtype: {compute_dtype}")
        logger.info(f"  Double quantization: {use_double_quant}")
    
    def create_quantization_config(self) -> BitsAndBytesConfig:
        """
        Create the BitsAndBytes quantization configuration.
        
        Returns:
            BitsAndBytesConfig object configured for NF4 quantization
        """
        skip_modules = [
            "vae",
            "model.vae",
            "vae.decoder",
            "vae.encoder",
            "autoencoder",
            "model.autoencoder",
            "autoencoder.decoder",
            "autoencoder.encoder",
            "patch_embed",
            "model.patch_embed",
            "final_layer",
            "model.final_layer",
            "time_embed",
            "model.time_embed",
            "time_embed_2",
            "model.time_embed_2",
            "timestep_emb",
            "model.timestep_emb",
            # Critical: exclude attention projections to prevent corruption
            "attn.q_proj",
            "attn.k_proj",
            "attn.v_proj",
            "attn.o_proj",
            "attn.qkv_proj",
            "attn.out_proj",
            "self_attn",
            "cross_attn",
            # === MoE Gate/Router (CRITICAL - float32 by design, controls expert routing) ===
            "mlp.gate",          # HunyuanTopKGate — routes tokens to experts
            # === Shared Expert (runs on ALL tokens, outsized quality impact) ===
            "shared_mlp",        # Shared MLP expert, processes every token
            # === Embeddings & Output Head ===
            "lm_head",           # Output projection
            "model.wte",         # Word token embedding
            "wte",               # Word token embedding (alternate path)
            "model.ln_f",        # Final RMSNorm
            "ln_f",              # Final RMSNorm (alternate path)
        ]

        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # NormalFloat4 - optimal for normal distributions
            bnb_4bit_use_double_quant=self.use_double_quant,  # Nested quantization
            bnb_4bit_compute_dtype=torch.bfloat16,  # MUST be bfloat16 for stable attention
            modules_to_not_convert=skip_modules,
            llm_int8_skip_modules=skip_modules,
        )
        
        logger.info("Created quantization config:")
        logger.info(f"  Quantization type: NF4")
        logger.info(f"  Double quantization: {self.use_double_quant}")
        logger.info(f"  Compute dtype: {self.compute_dtype}")
        
        return config
    
    def load_and_quantize(self) -> AutoModelForCausalLM:
        """
        Load the model and apply quantization.
        
        Returns:
            Quantized model ready for inference
            
        Raises:
            RuntimeError: If model loading or quantization fails
        """
        try:
            logger.info("Starting model quantization...")
            logger.info("This will take several minutes on first run...")
            
            # Create quantization config
            quant_config = self.create_quantization_config()
            
            # Load model with quantization
            # Note: trust_remote_code=True is required for HunyuanImage-3.0
            logger.info(f"Loading model from {self.model_path}...")
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quant_config,
                device_map=self.device_map,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,  # Base dtype for non-quantized parts
                attn_implementation="sdpa",  # Use SDPA since FlashAttention may not work on Blackwell yet
            )
            
            logger.info("Model loaded and quantized successfully!")
            logger.info(f"Model device map: {model.hf_device_map}")
            
            # Get memory usage
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(f"Peak GPU memory allocated: {mem_allocated:.2f} GB")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load and quantize model: {str(e)}")
            raise RuntimeError(f"Quantization failed: {str(e)}") from e
    
    def save_quantized_model(self, model: AutoModelForCausalLM) -> None:
        """
        Save the quantized model and configuration.
        
        Note: Due to bitsandbytes limitations, we save the original model files
        along with the quantization config for fast reloading.
        
        Args:
            model: The quantized model to save
        """
        try:
            logger.info(f"Saving quantized model to {self.output_path}...")
            
            # Save the model (this saves the base weights + config)
            # The quantization will be reapplied on load using the config
            model.save_pretrained(
                self.output_path,
                safe_serialization=True,  # Use safetensors format
            )
            
            # Save quantization metadata for easy loading
            quant_metadata = {
                "quantization_method": "bitsandbytes_nf4",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": self.use_double_quant,
                "bnb_4bit_compute_dtype": "torch.bfloat16",
                "expected_vram_gb": 45,  # Approximate VRAM usage (higher due to unquantized attn)
                "notes": "Load with BitsAndBytesConfig for NF4 quantization. Attention layers kept in full precision.",
                "attention_layers_quantized": False
            }
            
            metadata_path = self.output_path / "quantization_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(quant_metadata, f, indent=2)
            
            logger.info("Saved quantization metadata")
            self._copy_support_assets()

            logger.info(f"Model saved successfully to {self.output_path}")
            logger.info("\nTo load this model:")
            logger.info("  1. Use the same BitsAndBytesConfig")
            logger.info("  2. Set device_map='cuda:0' to avoid offload")
            logger.info("  3. Expected VRAM usage: ~42GB")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise RuntimeError(f"Save failed: {str(e)}") from e
    
    def _copy_support_assets(self) -> None:
        """Copy tokenizer and VAE artifacts that are not captured by save_pretrained."""
        source_root = Path(self.model_path)
        dest_root = self.output_path

        # Individual tokenizer files commonly referenced by ComfyUI
        tokenizer_files = [
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "tokenizer_wrapper.py",
            "tokenization_hunyuan.py",
        ]

        copied_any = False
        for filename in tokenizer_files:
            src_file = source_root / filename
            if src_file.exists():
                dest_file = dest_root / filename
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dest_file)
                logger.info("Copied tokenizer asset: %s", filename)
                copied_any = True

        if not copied_any:
            logger.info("No standalone tokenizer files detected; assuming embedded tokenizer")

        # Copy tokenizer or VAE directories if present
        for folder_name in ["tokenizer", "vae"]:
            src_dir = source_root / folder_name
            dest_dir = dest_root / folder_name
            if src_dir.exists():
                shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
                logger.info("Mirrored directory %s", src_dir.name)

    def create_loading_script(self) -> None:
        """
        Create a helper script for loading the quantized model.
        """
        loading_script = f'''"""
Quick loader for quantized HunyuanImage-3.0 model.
Generated automatically by hunyuan_quantize_nf4.py
"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def load_quantized_hunyuan(model_path="{self.output_path}"):
    """Load the NF4 quantized HunyuanImage-3.0 model."""
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant={self.use_double_quant},
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="cuda:0",  # Load entirely on GPU 0
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    
    # Load tokenizer
    model.load_tokenizer(model_path)
    
    return model

if __name__ == "__main__":
    print("Loading quantized model...")
    model = load_quantized_hunyuan()
    print("Model loaded successfully!")
    print(f"Device map: {{model.hf_device_map}}")
'''
        
        script_path = self.output_path / "load_quantized.py"
        with open(script_path, 'w') as f:
            f.write(loading_script)
        
        logger.info(f"Created loading helper script: {script_path}")
    
    def quantize_and_save(self) -> None:
        """
        Complete quantization workflow: load, quantize, and save.
        """
        logger.info("=" * 60)
        logger.info("Starting HunyuanImage-3.0 NF4 Quantization")
        logger.info("=" * 60)
        
        # Load and quantize
        model = self.load_and_quantize()
        
        # Save
        self.save_quantized_model(model)
        
        # Create helper script
        self.create_loading_script()
        
        logger.info("=" * 60)
        logger.info("Quantization complete!")
        logger.info("=" * 60)


def main():
    """Main entry point for the quantization script."""
    parser = argparse.ArgumentParser(
        description="Quantize HunyuanImage-3.0 to NF4 format for efficient GPU inference"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./HunyuanImage-3",
        help="Path to source model directory (default: ./HunyuanImage-3)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./HunyuanImage-3-NF4",
        help="Path to save quantized model (default: ./HunyuanImage-3-NF4)"
    )
    parser.add_argument(
        "--no-double-quant",
        action="store_true",
        help="Disable nested quantization (saves less memory)"
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Computation dtype (default: bfloat16)"
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device mapping strategy (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Convert compute dtype string to torch dtype
    compute_dtype = torch.bfloat16 if args.compute_dtype == "bfloat16" else torch.float16
    
    # Create quantizer
    quantizer = HunyuanQuantizer(
        model_path=args.model_path,
        output_path=args.output_path,
        compute_dtype=compute_dtype,
        use_double_quant=not args.no_double_quant,
        device_map=args.device_map
    )
    
    # Run quantization
    try:
        quantizer.quantize_and_save()
        logger.info("\n✓ Success! Your quantized model is ready for ComfyUI integration.")
    except Exception as e:
        logger.error(f"\n✗ Quantization failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
