"""
HunyuanImage-3.0 V2 Cache System

Simple cache for V2 unified node. Stores loaded models and their
associated managers (BlockSwapManager, SimpleVAEManager).

Author: Eric Hiss (GitHub: EricRollei)
License: Dual License (Non-Commercial and Commercial Use)
Copyright (c) 2025 Eric Hiss. All rights reserved.
"""

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import torch

try:
    from .hunyuan_device import get_device_manager
except ImportError:
    from hunyuan_device import get_device_manager

logger = logging.getLogger(__name__)


@dataclass
class CachedModel:
    """A cached model with its associated managers."""
    model: Any
    quant_type: str
    is_moveable: bool
    device: torch.device
    dtype: torch.dtype
    model_path: str
    
    # Associated managers
    block_swap_manager: Optional[Any] = None
    vae_manager: Optional[Any] = None
    
    # State tracking
    is_on_gpu: bool = True
    load_time: float = 0.0
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    
    # Config at load time
    blocks_to_swap: int = 0
    vae_placement: str = "always_gpu"
    
    # BF16-specific: track the VRAM reserve model was loaded with
    # If a new generation needs more reserve, cache must be invalidated
    loaded_with_reserve_gb: float = 0.0
    
    def touch(self):
        """Update last used time and increment use count."""
        self.last_used = time.time()
        self.use_count += 1
    
    def __repr__(self) -> str:
        return (
            f"CachedModel({self.quant_type}, "
            f"{'GPU' if self.is_on_gpu else 'CPU'}, "
            f"uses={self.use_count})"
        )


class ModelCacheV2:
    """
    Simple model cache for V2 unified node.
    
    Features:
    - Single model cache (one model at a time for simplicity)
    - Stores model + managers together
    - Soft unload support (move to CPU, keep in cache)
    - Full unload (remove from cache)
    - Thread-safe operations
    
    Usage:
        cache = ModelCacheV2()
        
        # Check if model is cached
        cached = cache.get(model_path, quant_type)
        if cached:
            return cached.model
        
        # Load and cache
        model = load_model(...)
        cache.put(model_path, quant_type, model, ...)
    """
    
    _instance: Optional['ModelCacheV2'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for global cache."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._cache: Dict[str, CachedModel] = {}
        self._cache_lock = threading.Lock()
        self._initialized = True
        
        logger.info("ModelCacheV2 initialized")
    
    def _make_key(self, model_path: str, quant_type: str) -> str:
        """Create cache key from path and quant type."""
        # Normalize path
        path = Path(model_path).resolve()
        return f"{path}::{quant_type}"
    
    def get(
        self,
        model_path: str,
        quant_type: str
    ) -> Optional[CachedModel]:
        """
        Get a cached model if it exists and is valid.
        
        Args:
            model_path: Path to model
            quant_type: Quantization type (bf16, int8, nf4)
            
        Returns:
            CachedModel if found and valid, None otherwise
        """
        key = self._make_key(model_path, quant_type)
        
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached:
                # CRITICAL: Validate the model is actually in memory
                # After a restart, cache entry may exist but model is gone
                try:
                    model = cached.model
                    if model is None:
                        logger.warning(f"Cache entry exists but model is None: {key}")
                        del self._cache[key]
                        return None
                    
                    # Check if model has any parameters and they're on a real device
                    try:
                        first_param = next(model.parameters())
                        device = first_param.device
                        
                        # If on meta device, model didn't load properly
                        if device.type == 'meta':
                            logger.warning(f"Cached model has meta tensors (not loaded): {key}")
                            del self._cache[key]
                            return None
                        
                        # Check if tensor data is accessible (not deallocated)
                        # This will throw if the storage is gone
                        _ = first_param.data_ptr()
                        
                    except (StopIteration, RuntimeError) as e:
                        logger.warning(f"Cached model appears invalid (no valid params): {key}, {e}")
                        del self._cache[key]
                        return None
                    
                    cached.touch()
                    logger.debug(f"Cache hit (validated): {key}")
                    return cached
                    
                except Exception as e:
                    logger.warning(f"Cache validation failed: {key}, {e}")
                    try:
                        del self._cache[key]
                    except:
                        pass
                    return None
        
        logger.debug(f"Cache miss: {key}")
        return None
    
    def put(
        self,
        model_path: str,
        quant_type: str,
        model: Any,
        is_moveable: bool,
        device: torch.device,
        dtype: torch.dtype,
        block_swap_manager: Optional[Any] = None,
        vae_manager: Optional[Any] = None,
        load_time: float = 0.0,
        blocks_to_swap: int = 0,
        vae_placement: str = "always_gpu",
        loaded_with_reserve_gb: float = 0.0
    ) -> CachedModel:
        """
        Add a model to the cache.
        
        If a different model is already cached, it will be unloaded first.
        
        Args:
            model_path: Path to model
            quant_type: Quantization type
            model: The loaded model
            is_moveable: Whether model can be moved between devices
            device: Current device
            dtype: Model dtype
            block_swap_manager: Optional BlockSwapManager
            vae_manager: Optional SimpleVAEManager
            load_time: Time taken to load
            blocks_to_swap: Number of blocks configured for swap
            vae_placement: VAE placement strategy
            loaded_with_reserve_gb: VRAM reserve the model was loaded with (for BF16)
            
        Returns:
            The CachedModel entry
        """
        key = self._make_key(model_path, quant_type)
        
        with self._cache_lock:
            # Check if same model already cached
            if key in self._cache:
                logger.info(f"Model already cached: {key}")
                cached = self._cache[key]
                cached.touch()
                return cached
            
            # Clear any existing cache entries (single model cache)
            if self._cache:
                self._clear_cache_internal()
            
            # Create new cache entry
            cached = CachedModel(
                model=model,
                quant_type=quant_type,
                is_moveable=is_moveable,
                device=device,
                dtype=dtype,
                model_path=model_path,
                block_swap_manager=block_swap_manager,
                vae_manager=vae_manager,
                is_on_gpu=True,
                load_time=load_time,
                blocks_to_swap=blocks_to_swap,
                vae_placement=vae_placement,
                loaded_with_reserve_gb=loaded_with_reserve_gb
            )
            
            self._cache[key] = cached
            logger.info(f"Cached model: {key}")
            
            return cached
    
    def soft_unload(self, model_path: str, quant_type: str) -> bool:
        """
        Soft unload a model (move to CPU, keep in cache).
        
        Only works for moveable models (NF4, BF16).
        INT8 models cannot be soft unloaded.
        
        Args:
            model_path: Path to model
            quant_type: Quantization type
            
        Returns:
            True if unloaded, False if not possible
        """
        key = self._make_key(model_path, quant_type)
        
        with self._cache_lock:
            cached = self._cache.get(key)
            if not cached:
                logger.warning(f"Cannot soft unload: model not in cache: {key}")
                return False
            
            if not cached.is_moveable:
                logger.warning(f"Cannot soft unload: model not moveable (INT8): {key}")
                return False
            
            if not cached.is_on_gpu:
                logger.info(f"Model already on CPU: {key}")
                return True
            
            # Use block swap manager if available
            if cached.block_swap_manager:
                cached.block_swap_manager.move_all_to_cpu()
            else:
                # Direct move
                cached.model.to("cpu")
            
            # Move VAE to CPU
            if cached.vae_manager:
                cached.vae_manager.cleanup_after_decode()

            cached.is_on_gpu = False
            get_device_manager().empty_cache()
            
            logger.info(f"Soft unloaded model: {key}")
            return True
    
    def restore(self, model_path: str, quant_type: str, device: str = "cuda:0") -> bool:
        """
        Restore a soft-unloaded model to GPU.
        
        Args:
            model_path: Path to model
            quant_type: Quantization type
            device: Target device
            
        Returns:
            True if restored, False if not possible
        """
        key = self._make_key(model_path, quant_type)
        
        with self._cache_lock:
            cached = self._cache.get(key)
            if not cached:
                logger.warning(f"Cannot restore: model not in cache: {key}")
                return False
            
            if cached.is_on_gpu:
                logger.info(f"Model already on GPU: {key}")
                return True
            
            # Use block swap manager if available
            if cached.block_swap_manager:
                cached.block_swap_manager.move_all_to_gpu()
            else:
                # Direct move
                cached.model.to(device)
            
            cached.is_on_gpu = True
            cached.device = torch.device(device)
            
            logger.info(f"Restored model to {device}: {key}")
            return True
    
    def full_unload(self, model_path: str = None, quant_type: str = None) -> bool:
        """
        Fully unload and remove a model from cache.
        
        If no arguments provided, clears entire cache.
        
        Args:
            model_path: Path to model (optional)
            quant_type: Quantization type (optional)
            
        Returns:
            True if unloaded
        """
        with self._cache_lock:
            if model_path and quant_type:
                key = self._make_key(model_path, quant_type)
                if key in self._cache:
                    self._unload_entry(key)
                    return True
                return False
            else:
                self._clear_cache_internal()
                return True
    
    def _unload_entry(self, key: str):
        """Unload a single cache entry (internal, must hold lock).
        
        Carefully breaks all circular references before deleting the model
        to ensure RAM is actually freed. Without this, BlockSwapManager's
        references to model and model.model.layers[] keep ~80GB+ of
        transformer block tensors alive in system RAM.
        """
        if key not in self._cache:
            return
        
        cached = self._cache[key]
        
        # Track RAM before cleanup
        try:
            import psutil
            ram_before = psutil.Process().memory_info().rss / 1024**3
        except ImportError:
            ram_before = None
        
        # Step 1: Clean up BlockSwapManager (MUST happen before del model)
        # cleanup() removes hooks, clears blocks list, nulls model ref
        if cached.block_swap_manager:
            if hasattr(cached.block_swap_manager, 'cleanup'):
                cached.block_swap_manager.cleanup()
            elif hasattr(cached.block_swap_manager, 'remove_hooks'):
                if cached.block_swap_manager.hooks_installed:
                    cached.block_swap_manager.remove_hooks()
            cached.block_swap_manager = None
        
        # Step 2: VAE manager
        if hasattr(cached, 'vae_manager') and cached.vae_manager is not None:
            if hasattr(cached.vae_manager, 'cleanup'):
                cached.vae_manager.cleanup()
            cached.vae_manager = None
        
        # Step 3: Unpatch VAE decode closure (holds ref to model)
        logger.info("  Step 3: Unpatching VAE decode closure...")
        if cached.model is not None:
            try:
                from .hunyuan_shared import unpatch_pipeline_pre_vae_cleanup
                unpatch_pipeline_pre_vae_cleanup(cached.model)
                logger.info("    Done: VAE decode closure unpatched")
            except ImportError:
                try:
                    from hunyuan_shared import unpatch_pipeline_pre_vae_cleanup
                    unpatch_pipeline_pre_vae_cleanup(cached.model)
                    logger.info("    Done: VAE decode closure unpatched (alt import)")
                except ImportError:
                    logger.info("    Skipped: unpatch_pipeline_pre_vae_cleanup not importable")
            except Exception as e:
                logger.warning(f"    Error unpatching VAE: {e}")
        
        # Step 4: Clear generation caches (KV cache, etc.)
        if cached.model is not None:
            try:
                from .hunyuan_shared import clear_generation_cache
                clear_generation_cache(cached.model)
            except ImportError:
                try:
                    from hunyuan_shared import clear_generation_cache
                    clear_generation_cache(cached.model)
                except ImportError:
                    pass
            except Exception:
                pass

        # Step 4b: Unpatch MoE efficient forward (releases _MOE_ORIGINAL_FORWARDS global dict)
        if cached.model is not None:
            try:
                try:
                    from .hunyuan_shared import unpatch_moe_efficient_forward
                except ImportError:
                    from hunyuan_shared import unpatch_moe_efficient_forward
                unpatch_moe_efficient_forward(cached.model)
                logger.info("  Step 4b: Unpatched MoE efficient forward")
            except ImportError:
                logger.info("  Step 4b: unpatch_moe_efficient_forward not available")
            except Exception as e:
                logger.warning(f"  Step 4b: Error unpatching MoE: {e}")
        
        # Step 4c: Reset dtype hooks flag so they get reinstalled on next load
        try:
            import sys
            _mod = sys.modules.get('hunyuan_shared') or sys.modules.get(__package__ + '.hunyuan_shared' if __package__ else 'hunyuan_shared')
            if _mod is not None:
                _mod._DTYPE_HOOKS_INSTALLED = False
                logger.info("  Step 4c: Reset _DTYPE_HOOKS_INSTALLED flag")
        except Exception:
            pass
        
        # Step 5: Remove accelerate hooks.
        # Also clean up instance-level `forward` attributes left by
        # remove_hook_from_module — they shadow the class method and
        # would be found by Step 5a's monkey-patch scanner, which could
        # (before the type guard fix) nuke their __class__ closure cell.
        if cached.model is not None:
            try:
                from accelerate.hooks import remove_hook_from_module
                for name, module in cached.model.named_modules():
                    if hasattr(module, '_hf_hook'):
                        remove_hook_from_module(module)
                    # Clean up stale instance-level forward left by hook removal
                    if 'forward' in vars(module):
                        try:
                            delattr(module, 'forward')
                        except Exception:
                            pass
            except (ImportError, Exception):
                pass
        
        # Step 5a: Remove external monkey-patches on model submodules.
        #
        # Other custom nodes (e.g. seedvr2_videoupscaler) may monkey-patch
        # methods on our model's submodules with closures that capture the
        # model or its children.  These closures keep the entire model tree
        # alive even after we delete our reference.
        #
        # Walk every submodule; for each instance attribute that is a
        # function/method with a __closure__, break the closure cells.
        # This is safe because we are about to delete the model anyway.
        logger.info("  Step 5a: Removing external monkey-patches on submodules...")
        if cached.model is not None:
            import types as _types
            import ctypes
            ext_closures_broken = 0
            try:
                for mod_name, submod in cached.model.named_modules():
                    # Check instance __dict__ for monkey-patched methods
                    for attr_name in list(vars(submod).keys()):
                        attr = vars(submod).get(attr_name)
                        if attr is None:
                            continue
                        closure = None
                        if isinstance(attr, _types.FunctionType):
                            closure = attr.__closure__
                        elif isinstance(attr, _types.MethodType):
                            closure = getattr(attr.__func__, '__closure__', None)
                        if closure:
                            for cell in closure:
                                try:
                                    _val = cell.cell_contents
                                except ValueError:
                                    continue
                                # CRITICAL: never nuke __class__ cells.
                                # Python stores the enclosing class in a
                                # closure cell for zero-arg super(); the
                                # cell lives on the class-level function
                                # object and is shared by ALL instances.
                                # Nuking it permanently breaks super()
                                # for the entire class (including future
                                # instances loaded via sys.modules).
                                if isinstance(_val, type):
                                    continue
                                try:
                                    ctypes.pythonapi.PyCell_Set(
                                        ctypes.py_object(cell),
                                        ctypes.py_object(None))
                                    ext_closures_broken += 1
                                except Exception:
                                    pass
                            # Remove the monkey-patched attr entirely
                            try:
                                delattr(submod, attr_name)
                            except Exception:
                                pass
            except Exception as e:
                logger.debug(f"    Error during external unpatch: {e}")
            logger.info(f"    5a: Broke {ext_closures_broken} closure cells "
                        f"from external monkey-patches")
        
        # Step 5b: Cleanly reverse all monkey-patches on the model.
        #
        # CRITICAL: We must RESTORE original methods, not just break closure
        # cells. Breaking cells leaves broken wrappers that corrupt the next
        # model load (the new load wraps the broken wrapper, creating a
        # corruption cascade that kills super() and pipeline.__call__).
        logger.info("  Step 5b: Restoring original methods (unpatching)...")
        if cached.model is not None:
            model_id = id(cached.model)
            logger.info(f"    Model object id: {model_id:#x}")
            
            # --- 5b.1+2+3: Unpatch generate_image and pipeline.__call__ ---
            # Uses saved originals (_comfy_original_generate_image, _comfy_original_call)
            try:
                try:
                    from .hunyuan_shared import unpatch_hunyuan_generate_image
                except ImportError:
                    from hunyuan_shared import unpatch_hunyuan_generate_image
                unpatch_hunyuan_generate_image(cached.model)
                logger.info("    5b.1-3: Unpatched generate_image + pipeline.__call__")
            except ImportError:
                logger.warning("    5b.1-3: unpatch_hunyuan_generate_image not available")
            except Exception as e:
                logger.warning(f"    5b.1-3: Error during unpatch: {e}")
            
            # --- 5b.4: Break VAE decode closure (backup) ---
            # Step 3 already called unpatch_pipeline_pre_vae_cleanup to restore
            # original decode, but the decode_with_cleanup function object may
            # survive in gc. Force-break any remaining VAE closure cells.
            pipeline = getattr(cached.model, 'pipeline', None)
            vae = None
            if pipeline is not None:
                vae = getattr(pipeline, 'vae', None)
            if vae is None:
                vae = getattr(cached.model, 'vae', None)
            if vae is not None:
                # If decode_with_cleanup is still around as _original_decode
                # or in some other attribute, break its closure
                for vae_attr in ('decode', '_original_decode',
                                 '_decode', '_original_forward'):
                    fn = getattr(vae, vae_attr, None)
                    if fn is not None and hasattr(fn, '__closure__') and fn.__closure__:
                        import ctypes
                        for cell in fn.__closure__:
                            try:
                                ctypes.pythonapi.PyCell_Set(
                                    ctypes.py_object(cell),
                                    ctypes.py_object(None))
                            except Exception:
                                pass
                        logger.info(f"    5b.4: Cleared closure cells on "
                                    f"vae.{vae_attr}")
                # Also remove our patch markers
                if hasattr(vae, '_prevae_cleanup_patched'):
                    vae._prevae_cleanup_patched = False
                if hasattr(vae, '_original_decode'):
                    try:
                        del vae._original_decode
                    except Exception:
                        pass
            else:
                logger.info("    5b.4: No VAE found")
            
            # --- 5b.5: NUCLEAR — walk ALL gc cells referencing this model ---
            # After all targeted cleanup, find ANY remaining closure cell
            # that still holds a reference to the model (directly or via
            # its submodules) and break it.
            import gc
            import ctypes
            _cell_sentinel = None
            cell_type = type((lambda: _cell_sentinel).__closure__[0])
            gc.collect()
            
            # Pre-build set of all module ids for O(1) lookup
            # Also include pipeline and its submodules
            model_module_ids = set()
            try:
                for _, submod in cached.model.named_modules():
                    model_module_ids.add(id(submod))
            except Exception:
                pass
            model_module_ids.add(id(cached.model))
            if pipeline is not None:
                model_module_ids.add(id(pipeline))
                # Pipeline may have its own submodules
                try:
                    for _, submod in pipeline.named_modules():
                        model_module_ids.add(id(submod))
                except Exception:
                    pass
            
            nuked = 0
            for obj in gc.get_objects():
                if type(obj) is not cell_type:
                    continue
                try:
                    val = obj.cell_contents
                except ValueError:
                    continue
                # Check if this cell holds our model, pipeline, or any
                # of their submodules.
                # CRITICAL: skip class objects (types) — Python stores __class__
                # in a closure cell for super(); breaking it kills super() globally.
                if isinstance(val, torch.nn.Module) and not isinstance(val, type) and id(val) in model_module_ids:
                    ctypes.pythonapi.PyCell_Set(
                        ctypes.py_object(obj), ctypes.py_object(None))
                    nuked += 1
                # Also check for callable capturing model (e.g. bound methods,
                # partial objects holding the model)
                elif callable(val) and hasattr(val, '__self__'):
                    if id(getattr(val, '__self__', None)) in model_module_ids:
                        ctypes.pythonapi.PyCell_Set(
                            ctypes.py_object(obj), ctypes.py_object(None))
                        nuked += 1
            del model_module_ids
            logger.info(f"    5b.5: Nuclear cell scan broke {nuked} closure cells")
        
        # Step 6: Clear attached metadata and free tensor storage in-place
        logger.info("  Step 6: Clearing metadata and freeing tensor storage...")
        
        # Clean up local variables from earlier steps that hold submodule refs
        # (pipeline, vae were set in Step 5b.4 and prevent gc of model)
        try:
            del pipeline
        except (NameError, UnboundLocalError):
            pass
        try:
            del vae
        except (NameError, UnboundLocalError):
            pass
        
        if cached.model is not None:
            for attr in ('_hunyuan_info', '_hunyuan_path', '_block_swap_manager'):
                if hasattr(cached.model, attr):
                    try:
                        setattr(cached.model, attr, None)
                    except Exception:
                        pass
            
            if cached.blocks_to_swap > 0:
                # Block swap was active: free all tensor storage IN-PLACE.
                # This releases GPU memory and pinned CPU buffers immediately
                # without creating new CRT heap allocations (no model.to('cpu')).
                #
                # Even if the model object survives due to stale references
                # (refcount > 1), the actual tensor data (~150GB) is freed.
                logger.info("    Block swap active: freeing tensor storage in-place...")
                freed_gpu_bytes = 0
                freed_cpu_bytes = 0
                empty_cpu = torch.empty(0, device='cpu')
                
                try:
                    # Free ALL model parameters (blocks + non-block components)
                    for name, param in cached.model.named_parameters():
                        nbytes = param.data.numel() * param.data.element_size()
                        if param.data.device.type == 'cuda':
                            freed_gpu_bytes += nbytes
                        else:
                            freed_cpu_bytes += nbytes
                        param.data = empty_cpu
                    
                    # Free ALL model buffers (registered buffers, embeddings, etc.)
                    for name, buf in cached.model.named_buffers():
                        nbytes = buf.data.numel() * buf.data.element_size()
                        if buf.data.device.type == 'cuda':
                            freed_gpu_bytes += nbytes
                        else:
                            freed_cpu_bytes += nbytes
                        buf.data = empty_cpu
                    
                    # Also gut the VAE (may be parked on CPU with large tensors)
                    for vae_path in ('vae', 'pipeline.vae'):
                        vae_obj = cached.model
                        try:
                            for part in vae_path.split('.'):
                                vae_obj = getattr(vae_obj, part)
                            for name, param in vae_obj.named_parameters():
                                nbytes = param.data.numel() * param.data.element_size()
                                if param.data.device.type == 'cuda':
                                    freed_gpu_bytes += nbytes
                                else:
                                    freed_cpu_bytes += nbytes
                                param.data = empty_cpu
                            for name, buf in vae_obj.named_buffers():
                                nbytes = buf.data.numel() * buf.data.element_size()
                                if buf.data.device.type == 'cuda':
                                    freed_gpu_bytes += nbytes
                                else:
                                    freed_cpu_bytes += nbytes
                                buf.data = empty_cpu
                        except (AttributeError, StopIteration):
                            pass
                    
                    del empty_cpu
                    
                    logger.info(f"    Freed {freed_gpu_bytes/1024**3:.1f}GB GPU + "
                               f"{freed_cpu_bytes/1024**3:.1f}GB CPU tensor storage")
                    
                except Exception as e:
                    logger.warning(f"    Tensor cleanup error: {e}")

                # Flush device caching allocator to return freed GPU memory
                device_manager = get_device_manager()
                device_manager.empty_cache()

                # Log memory after cleanup (CUDA only)
                if device_manager.device_type.value == "cuda":
                    try:
                        vram_after = torch.cuda.memory_allocated() / 1024**3
                        vram_free = (torch.cuda.get_device_properties(0).total_memory
                                    - torch.cuda.memory_reserved()) / 1024**3
                        logger.info(f"    After tensor cleanup: {vram_after:.1f}GB VRAM allocated, "
                                   f"~{vram_free:.1f}GB free")
                    except Exception:
                        pass
                else:
                    logger.info(f"    After tensor cleanup: (memory tracking not available on this device)")
                logger.info(f"    After tensor cleanup: {vram_after:.1f}GB VRAM allocated, "
                           f"~{vram_free:.1f}GB free")
            else:
                # No block swap: standard move to CPU before delete
                try:
                    if hasattr(cached.model, 'to') and not hasattr(cached.model, 'hf_device_map'):
                        cached.model.to('cpu')
                        logger.info("    Moved model to CPU")
                except Exception as e:
                    logger.warning(f"    Failed to move to CPU: {e}")
        
        # Step 7: Delete model and cache entry
        logger.info("  Step 7: Deleting model reference...")
        model_ref_count = sys.getrefcount(cached.model) if cached.model is not None else 0
        logger.info(f"    Model refcount before del: {model_ref_count}")
        del cached.model
        del self._cache[key]
        
        # Step 8: Aggressive garbage collection
        logger.info("  Step 8: Garbage collection...")
        import gc
        gc.collect()
        gc.collect()  # second pass catches garbage created by __del__ finalizers
        get_device_manager().empty_cache()
        
        # Step 8b removed — the scoped nuclear scan in Step 5b.5 already
        # broke all closure cells referencing Hunyuan modules.  Any nn.Module
        # surviving here belongs to downstream models (Marigold, segmentation, etc.)
        
        # Step 9: Clear PyTorch internal allocator caches
        logger.info("  Step 9: Clearing allocator caches...")
        device_manager = get_device_manager()
        device_manager.empty_cache()
        # Clear CUDA host (pinned memory) allocator cache (CUDA only)
        try:
            torch._C._host_emptyCache()
            logger.info("    Cleared CUDA host allocator cache")
        except Exception:
            pass
        # Clear accelerator caches
        try:
            torch._C._accelerator_emptyCache()
            logger.info("    Cleared accelerator cache")
        except Exception:
            pass
        
        # Step 10: Force Windows to return freed memory to OS
        logger.info("  Step 10: Windows memory release...")
        try:
            from .hunyuan_shared import force_windows_memory_release
        except ImportError:
            from hunyuan_shared import force_windows_memory_release
        force_windows_memory_release()
        
        # Report RAM freed
        if ram_before is not None:
            try:
                ram_after = psutil.Process().memory_info().rss / 1024**3
                ram_freed = ram_before - ram_after
                logger.info(f"Fully unloaded: {key} "
                           f"(RAM freed: {ram_freed:.1f}GB, current: {ram_after:.1f}GB)")
            except Exception:
                logger.info(f"Fully unloaded: {key}")
        else:
            logger.info(f"Fully unloaded: {key}")
    
    def _clear_cache_internal(self):
        """Clear entire cache (internal, must hold lock)."""
        for key in list(self._cache.keys()):
            self._unload_entry(key)
        self._cache.clear()
    
    def get_status(self) -> Dict:
        """Get cache status information."""
        with self._cache_lock:
            if not self._cache:
                return {"cached": False}
            
            # Get first (only) entry
            key, cached = next(iter(self._cache.items()))
            
            return {
                "cached": True,
                "model_path": cached.model_path,
                "quant_type": cached.quant_type,
                "is_on_gpu": cached.is_on_gpu,
                "is_moveable": cached.is_moveable,
                "use_count": cached.use_count,
                "blocks_to_swap": cached.blocks_to_swap,
                "vae_placement": cached.vae_placement,
                "load_time": cached.load_time,
            }
    
    def clear(self):
        """Clear entire cache."""
        self.full_unload()


# Global cache instance
_cache: Optional[ModelCacheV2] = None


def get_cache() -> ModelCacheV2:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        _cache = ModelCacheV2()
    return _cache
