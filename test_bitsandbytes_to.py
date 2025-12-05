"""
Test script to verify bitsandbytes 0.48.2 .to() device movement support.

This tests whether we can move a quantized model between GPU and CPU
without reloading from disk - the key feature for making Soft Unload work.

Run with ComfyUI's Python:
  A:\Comfy25\ComfyUI_windows_portable\python_embeded\python.exe test_bitsandbytes_to.py
"""

import time
import gc
import torch

def format_bytes(b):
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if b < 1024:
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} TB"

def get_vram_info():
    """Get current VRAM usage."""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        used = total - free
        return used, free, total
    return 0, 0, 0

def print_vram(label=""):
    """Print current VRAM state."""
    used, free, total = get_vram_info()
    print(f"  VRAM {label}: {format_bytes(used)} used / {format_bytes(total)} total ({format_bytes(free)} free)")

def test_linear_layers():
    """Test .to() on individual Linear8bitLt and Linear4bit layers."""
    print("\n" + "="*60)
    print("TEST 1: Individual Linear Layer Movement")
    print("="*60)
    
    try:
        from bitsandbytes.nn import Linear8bitLt, Linear4bit
        import bitsandbytes as bnb
        print(f"bitsandbytes version: {bnb.__version__}")
    except ImportError as e:
        print(f"ERROR: Could not import bitsandbytes: {e}")
        return False
    
    # Test Linear4bit (NF4)
    print("\n--- Testing Linear4bit (NF4) ---")
    try:
        # Create a small 4-bit linear layer on GPU
        layer_4bit = Linear4bit(
            input_features=1024, 
            output_features=1024,
            bias=False,
            compute_dtype=torch.bfloat16,
            quant_type='nf4'
        ).cuda()
        
        # Initialize with some weights
        with torch.no_grad():
            layer_4bit.weight = torch.nn.Parameter(
                torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
            )
        
        print(f"  Created Linear4bit on: {next(layer_4bit.parameters()).device}")
        print_vram("after creation")
        
        # Try moving to CPU
        print("  Moving to CPU...")
        start = time.time()
        layer_4bit = layer_4bit.to('cpu')
        cpu_time = time.time() - start
        print(f"  Now on: {next(layer_4bit.parameters()).device}")
        print(f"  Move to CPU took: {cpu_time:.3f}s")
        print_vram("after CPU move")
        
        # Try moving back to GPU
        print("  Moving back to GPU...")
        start = time.time()
        layer_4bit = layer_4bit.to('cuda')
        gpu_time = time.time() - start
        print(f"  Now on: {next(layer_4bit.parameters()).device}")
        print(f"  Move to GPU took: {gpu_time:.3f}s")
        print_vram("after GPU move")
        
        # Test inference still works
        print("  Testing inference...")
        test_input = torch.randn(1, 1024, dtype=torch.bfloat16, device='cuda')
        output = layer_4bit(test_input)
        print(f"  Inference output shape: {output.shape} ✓")
        
        del layer_4bit, test_input, output
        torch.cuda.empty_cache()
        gc.collect()
        
        print("  Linear4bit .to() test: PASSED ✓")
        return True
        
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return False


def test_model_level():
    """Test .to() on a full model with quantized layers."""
    print("\n" + "="*60)
    print("TEST 2: Full Model Movement (Small Test Model)")
    print("="*60)
    
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Linear4bit
    except ImportError as e:
        print(f"ERROR: Could not import bitsandbytes: {e}")
        return False
    
    # Create a small model with quantized layers
    class SmallQuantizedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = Linear4bit(512, 512, bias=False, compute_dtype=torch.bfloat16, quant_type='nf4')
            self.layer2 = Linear4bit(512, 512, bias=False, compute_dtype=torch.bfloat16, quant_type='nf4')
            self.norm = torch.nn.LayerNorm(512)
        
        def forward(self, x):
            x = self.layer1(x)
            x = torch.nn.functional.gelu(x)
            x = self.layer2(x)
            x = self.norm(x)
            return x
    
    try:
        print("  Creating small quantized model on GPU...")
        model = SmallQuantizedModel().cuda()
        print_vram("after model creation")
        
        # Try model.to('cpu')
        print("  Moving entire model to CPU...")
        start = time.time()
        model = model.to('cpu')
        cpu_time = time.time() - start
        print(f"  Move to CPU took: {cpu_time:.3f}s")
        print_vram("after CPU move")
        
        # Try model.to('cuda')
        print("  Moving entire model back to GPU...")
        start = time.time()
        model = model.to('cuda')
        gpu_time = time.time() - start
        print(f"  Move to GPU took: {gpu_time:.3f}s")
        print_vram("after GPU move")
        
        # Test inference
        print("  Testing inference after round-trip...")
        test_input = torch.randn(1, 512, dtype=torch.bfloat16, device='cuda')
        output = model(test_input)
        print(f"  Inference output shape: {output.shape} ✓")
        
        del model, test_input, output
        torch.cuda.empty_cache()
        gc.collect()
        
        print("  Full model .to() test: PASSED ✓")
        return True
        
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hunyuan_model():
    """Test .to() on actual loaded Hunyuan NF4 model (if available in cache)."""
    print("\n" + "="*60)
    print("TEST 3: Real Hunyuan Model Movement (if cached)")
    print("="*60)
    
    try:
        # Try to import our cache
        import sys
        sys.path.insert(0, r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\custom_nodes\Eric_Hunyuan3")
        from hunyuan_shared import HunyuanModelCache
        
        if HunyuanModelCache._cached_model is None:
            print("  No Hunyuan model in cache. Skipping this test.")
            print("  (Load a model in ComfyUI first, then run this script)")
            return None
        
        model = HunyuanModelCache._cached_model
        print(f"  Found cached model: {HunyuanModelCache._cached_path}")
        print_vram("before move")
        
        # Check current device
        try:
            sample_param = next(model.parameters())
            print(f"  Current device: {sample_param.device}")
        except StopIteration:
            print("  Warning: Could not get sample parameter")
        
        # Try moving to CPU
        print("\n  Attempting model.to('cpu')...")
        start = time.time()
        try:
            model.to('cpu')
            cpu_time = time.time() - start
            print(f"  Move to CPU took: {cpu_time:.2f}s")
            print_vram("after CPU move")
            
            # Force CUDA cache clear
            torch.cuda.empty_cache()
            gc.collect()
            print_vram("after cache clear")
            
            # Try moving back
            print("\n  Attempting model.to('cuda')...")
            start = time.time()
            model.to('cuda')
            gpu_time = time.time() - start
            print(f"  Move to GPU took: {gpu_time:.2f}s")
            print_vram("after GPU move")
            
            print("\n  Real Hunyuan model .to() test: PASSED ✓")
            print(f"  Round-trip time: {cpu_time + gpu_time:.2f}s (vs ~2min reload from disk)")
            return True
            
        except Exception as e:
            print(f"  ERROR during move: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except ImportError as e:
        print(f"  Could not import HunyuanModelCache: {e}")
        return None


def main():
    print("="*60)
    print("BITSANDBYTES .to() DEVICE MOVEMENT TEST")
    print("="*60)
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print_vram("initial")
    
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes version: {bnb.__version__}")
    except ImportError:
        print("ERROR: bitsandbytes not installed!")
        return
    
    results = {}
    
    # Test 1: Individual layers
    results['linear_layers'] = test_linear_layers()
    
    # Test 2: Small model
    results['small_model'] = test_model_level()
    
    # Test 3: Real Hunyuan (if cached)
    results['hunyuan'] = test_hunyuan_model()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "PASSED ✓"
        else:
            status = "FAILED ✗"
        print(f"  {test}: {status}")
    
    if all(r for r in results.values() if r is not None):
        print("\n✅ All tests passed! Soft Unload should work with bitsandbytes 0.48.2")
    else:
        print("\n⚠️ Some tests failed. Check errors above.")


if __name__ == "__main__":
    main()
