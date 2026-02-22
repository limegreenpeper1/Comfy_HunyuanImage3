"""
HunyuanImage-3.0 Device Abstraction Layer - MPS Support

Provides device-agnostic abstraction for CUDA, MPS (Apple Silicon), and CPU.
Enables BF16 model inference on Apple Silicon Macs with unified memory.

Author: Eric Hiss (GitHub: EricRollei)
License: Dual License (Non-Commercial and Commercial Use)
Copyright (c) 2025 Eric Hiss. All rights reserved.
"""

import logging
from enum import Enum
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported device types."""
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class DeviceManager:
    """
    Centralized device manager for CUDA, MPS, and CPU devices.

    Provides device-agnostic APIs for:
    - Device detection (MPS > CUDA > CPU)
    - Memory queries
    - Synchronization
    - Cache management

    Singleton pattern - use get_instance() to access.
    """

    _instance: Optional['DeviceManager'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._device_type: DeviceType = self._detect_device()
        self._mps_available: bool = self._check_mps_available()
        self._cuda_available: bool = torch.cuda.is_available()
        self._device_index: int = 0

        self._initialized = True
        logger.info(f"DeviceManager initialized: primary device = {self._device_type.value}")

    @staticmethod
    def get_instance() -> 'DeviceManager':
        """Get the singleton DeviceManager instance."""
        if DeviceManager._instance is None:
            DeviceManager._instance = DeviceManager()
        return DeviceManager._instance

    def _detect_device(self) -> DeviceType:
        """
        Detect the best available device.

        Priority: MPS > CUDA > CPU
        - MPS: Apple Silicon (M1/M2/M3/M4)
        - CUDA: NVIDIA GPUs
        - CPU: Fallback
        """
        # Check MPS first (Apple Silicon)
        if self._check_mps_available():
            return DeviceType.MPS

        # Then CUDA (NVIDIA)
        if torch.cuda.is_available():
            return DeviceType.CUDA

        # Fallback to CPU
        return DeviceType.CPU

    def _check_mps_available(self) -> bool:
        """
        Check if MPS (Metal Performance Shaders) is available.

        MPS requires:
        - Apple Silicon hardware (M1/M2/M3/M4)
        - PyTorch with MPS support
        - macOS 12.3+
        """
        if not hasattr(torch.backends, 'mps'):
            return False

        if not torch.backends.mps.is_available():
            return False

        # Check if MPS is built (not just available)
        if not torch.backends.mps.is_built():
            logger.warning("MPS is available but not built - falling back to CPU")
            return False

        return True

    @property
    def device_type(self) -> DeviceType:
        """Get the detected device type."""
        return self._device_type

    @property
    def is_mps_available(self) -> bool:
        """Check if MPS is available."""
        return self._mps_available

    @property
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self._cuda_available

    def get_device_string(self, device_idx: int = 0) -> str:
        """
        Get the device string for the specified device index.

        Args:
            device_idx: Device index (default: 0)

        Returns:
            Device string (e.g., "cuda:0", "mps", "cpu")
        """
        if self._device_type == DeviceType.MPS:
            return "mps"  # MPS doesn't use device indices
        elif self._device_type == DeviceType.CUDA:
            return f"cuda:{device_idx}"
        else:
            return "cpu"

    def parse_device(self, device_str: str) -> torch.device:
        """
        Parse a device string and return a torch.device.

        Args:
            device_str: Device string (e.g., "cuda:0", "mps", "cpu")

        Returns:
            torch.device object
        """
        return torch.device(device_str)

    def get_memory_info(self, device_idx: int = 0) -> Tuple[int, int]:
        """
        Get memory information for the device.

        For CUDA: returns (free_bytes, total_bytes) from torch.cuda.mem_get_info
        For MPS: estimates available system memory (80% of available)
        For CPU: returns system memory info

        Args:
            device_idx: Device index (used for CUDA only)

        Returns:
            Tuple of (free_bytes, total_bytes)
        """
        if self._device_type == DeviceType.CUDA and self._cuda_available:
            try:
                free, total = torch.cuda.mem_get_info(device_idx)
                return free, total
            except Exception as e:
                logger.warning(f"Failed to get CUDA memory info: {e}")
                return 0, 0

        elif self._device_type == DeviceType.MPS:
            # MPS doesn't provide memory query APIs
            # Use psutil to estimate available system memory
            try:
                import psutil
                mem = psutil.virtual_memory()
                # Use 80% of available as safe estimate
                available = int(mem.available * 0.8)
                return available, mem.total
            except ImportError:
                logger.warning("psutil not available - cannot get MPS memory info")
                return 0, 0

        else:  # CPU
            try:
                import psutil
                mem = psutil.virtual_memory()
                return mem.available, mem.total
            except ImportError:
                return 0, 0

    def get_allocated_memory(self, device_idx: int = 0) -> int:
        """
        Get allocated memory for the device.

        Args:
            device_idx: Device index (used for CUDA only)

        Returns:
            Allocated memory in bytes
        """
        if self._device_type == DeviceType.CUDA and self._cuda_available:
            try:
                return torch.cuda.memory_allocated(device_idx)
            except Exception:
                return 0

        # MPS and CPU don't expose allocated memory
        return 0

    def get_free_memory(self, device_idx: int = 0) -> int:
        """
        Get free memory for the device.

        Args:
            device_idx: Device index (used for CUDA only)

        Returns:
            Free memory in bytes
        """
        free, _ = self.get_memory_info(device_idx)
        return free

    def synchronize(self, device: Optional[torch.device] = None) -> None:
        """
        Synchronize operations on the device.

        Args:
            device: Optional device to synchronize. If None, uses primary device.
        """
        target_device = device or torch.device(self.get_device_string())

        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        elif target_device.type == "mps":
            # MPS synchronization may not be available in all PyTorch versions
            # Check if synchronize exists before calling
            if hasattr(torch.backends.mps, 'synchronize'):
                torch.backends.mps.synchronize()
            # Otherwise, MPS synchronizes automatically
        # CPU doesn't need synchronization

    def empty_cache(self) -> None:
        """
        Empty the device's memory cache.

        For CUDA: calls torch.cuda.empty_cache()
        For MPS: no-op (PyTorch handles automatically)
        For CPU: no-op
        """
        if self._device_type == DeviceType.CUDA and self._cuda_available:
            torch.cuda.empty_cache()
        # MPS and CPU don't have manual cache emptying

    def get_device_name(self, device_idx: int = 0) -> str:
        """
        Get the device name.

        Args:
            device_idx: Device index (used for CUDA only)

        Returns:
            Device name string
        """
        if self._device_type == DeviceType.CUDA and self._cuda_available:
            try:
                return torch.cuda.get_device_name(device_idx)
            except Exception:
                return "CUDA Device"
        elif self._device_type == DeviceType.MPS:
            return "Apple Silicon (MPS)"
        else:
            return "CPU"

    def set_device_index(self, device_idx: int) -> None:
        """
        Set the device index (for CUDA multi-GPU setups).

        Args:
            device_idx: Device index to use
        """
        self._device_index = device_idx

    def get_device_index(self) -> int:
        """Get the current device index."""
        return self._device_index

    def supports_bfloat16(self) -> bool:
        """
        Check if the device supports bfloat16.

        Returns:
            True if bfloat16 is supported, False otherwise
        """
        if self._device_type == DeviceType.CUDA:
            # Most modern CUDA GPUs support bfloat16
            return True
        elif self._device_type == DeviceType.MPS:
            # MPS does NOT support bfloat16 (as of 2025)
            return False
        else:
            # CPU may support bfloat16 depending on the CPU
            # For safety, we return False
            return False

    def get_recommended_dtype(self) -> torch.dtype:
        """
        Get the recommended dtype for the current device.

        Returns:
            torch.bfloat16 if supported, otherwise torch.float32
        """
        if self.supports_bfloat16():
            return torch.bfloat16
        else:
            return torch.float32


# Global convenience functions
def get_device_manager() -> DeviceManager:
    """Get the singleton DeviceManager instance."""
    return DeviceManager.get_instance()


def get_device_string(device_idx: int = 0) -> str:
    """Get the device string for the specified device index."""
    return get_device_manager().get_device_string(device_idx)


def get_device_type() -> DeviceType:
    """Get the detected device type."""
    return get_device_manager().device_type


def is_mps_available() -> bool:
    """Check if MPS is available."""
    return get_device_manager().is_mps_available


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return get_device_manager().is_cuda_available
