"""GPU utility module for device management and memory handling.

This module provides utilities for:
- Automatic CUDA device detection
- GPU memory management
- Out-of-memory (OOM) error handling
- CPU fallback support
"""

import logging
from typing import Optional, Callable, Any
from functools import wraps
import torch

logger = logging.getLogger(__name__)


class GPUManager:
    """Manager for GPU resources and memory.

    Provides centralized GPU device management with automatic detection,
    memory monitoring, and graceful fallback to CPU.
    """

    def __init__(self, device: Optional[str] = None, allow_cpu_fallback: bool = True):
        """Initialize GPU manager.

        Args:
            device: Specific device to use ("cuda", "cuda:0", "cpu", etc.)
                   If None, automatically detects best available device
            allow_cpu_fallback: Whether to fall back to CPU on GPU errors
        """
        self.allow_cpu_fallback = allow_cpu_fallback
        self._device = self._initialize_device(device)
        self._log_device_info()

    def _initialize_device(self, device: Optional[str]) -> torch.device:
        """Initialize and validate device.

        Args:
            device: Device string or None for auto-detection

        Returns:
            torch.device object
        """
        if device is not None:
            # Use specified device
            try:
                dev = torch.device(device)
                if dev.type == "cuda" and not torch.cuda.is_available():
                    logger.warning(
                        f"CUDA device '{device}' requested but CUDA not available. "
                        f"Falling back to CPU."
                    )
                    return torch.device("cpu")
                return dev
            except Exception as e:
                logger.error(f"Invalid device '{device}': {e}. Using CPU.")
                return torch.device("cpu")

        # Auto-detect best device
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            logger.info("CUDA not available. Using CPU.")
            return torch.device("cpu")

    def _log_device_info(self) -> None:
        """Log information about the selected device."""
        if self._device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(self._device)
            gpu_memory = torch.cuda.get_device_properties(self._device).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            logger.info(
                f"Using GPU: {gpu_name} "
                f"(Device {self._device.index if self._device.index is not None else 0}, "
                f"{gpu_memory_gb:.2f} GB)"
            )
        else:
            logger.info("Using CPU for computation")

    @property
    def device(self) -> torch.device:
        """Get the current device.

        Returns:
            torch.device object
        """
        return self._device

    def is_gpu_available(self) -> bool:
        """Check if GPU is available and being used.

        Returns:
            True if using GPU, False otherwise
        """
        return self._device.type == "cuda"

    def get_memory_info(self) -> dict:
        """Get GPU memory information.

        Returns:
            Dictionary with memory stats (empty if using CPU)
        """
        if not self.is_gpu_available():
            return {}

        allocated = torch.cuda.memory_allocated(self._device)
        reserved = torch.cuda.memory_reserved(self._device)
        total = torch.cuda.get_device_properties(self._device).total_memory

        return {
            "allocated_mb": allocated / (1024**2),
            "reserved_mb": reserved / (1024**2),
            "total_mb": total / (1024**2),
            "free_mb": (total - reserved) / (1024**2),
            "utilization": reserved / total if total > 0 else 0,
        }

    def clear_cache(self) -> None:
        """Clear GPU cache to free memory."""
        if self.is_gpu_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")

    def reset_peak_memory_stats(self) -> None:
        """Reset peak memory statistics."""
        if self.is_gpu_available():
            torch.cuda.reset_peak_memory_stats(self._device)

    def log_memory_usage(self, prefix: str = "") -> None:
        """Log current GPU memory usage.

        Args:
            prefix: Optional prefix for log message
        """
        if not self.is_gpu_available():
            return

        info = self.get_memory_info()
        msg = (
            f"{prefix}GPU Memory: "
            f"{info['allocated_mb']:.1f}MB allocated, "
            f"{info['reserved_mb']:.1f}MB reserved, "
            f"{info['free_mb']:.1f}MB free "
            f"({info['utilization']*100:.1f}% utilization)"
        )
        logger.info(msg)


def get_device(device: Optional[str] = None) -> torch.device:
    """Get torch device with automatic CUDA detection.

    Convenience function for quick device selection without creating a GPUManager.

    Args:
        device: Specific device string or None for auto-detection

    Returns:
        torch.device object

    Example:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device("cuda:1")  # Specific GPU
        >>> device = get_device("cpu")  # Force CPU
    """
    if device is not None:
        try:
            dev = torch.device(device)
            if dev.type == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Using CPU.")
                return torch.device("cpu")
            return dev
        except Exception as e:
            logger.error(f"Invalid device '{device}': {e}. Using CPU.")
            return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def handle_oom(
    fallback_fn: Optional[Callable] = None,
    clear_cache: bool = True,
    retry_on_cpu: bool = True,
) -> Callable:
    """Decorator to handle CUDA out-of-memory errors.

    Provides automatic handling of OOM errors with optional fallback strategies:
    - Clear GPU cache and retry
    - Fall back to CPU
    - Call custom fallback function

    Args:
        fallback_fn: Optional custom fallback function to call on OOM
        clear_cache: Whether to clear GPU cache before retrying
        retry_on_cpu: Whether to retry on CPU after OOM

    Returns:
        Decorated function with OOM handling

    Example:
        >>> @handle_oom(retry_on_cpu=True)
        ... def process_batch(data, device):
        ...     return model(data.to(device))
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"CUDA OOM in {func.__name__}: {e}")

                    # Clear cache if requested
                    if clear_cache and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("Cleared GPU cache")

                    # Try fallback function
                    if fallback_fn is not None:
                        logger.info(f"Calling fallback function for {func.__name__}")
                        return fallback_fn(*args, **kwargs)

                    # Try CPU fallback
                    if retry_on_cpu:
                        logger.info(f"Retrying {func.__name__} on CPU")
                        # Try to move tensors to CPU in kwargs
                        cpu_kwargs = {}
                        for k, v in kwargs.items():
                            if isinstance(v, torch.Tensor):
                                cpu_kwargs[k] = v.cpu()
                            elif k == "device":
                                cpu_kwargs[k] = torch.device("cpu")
                            else:
                                cpu_kwargs[k] = v

                        try:
                            return func(*args, **cpu_kwargs)
                        except Exception as cpu_error:
                            logger.error(f"CPU fallback also failed: {cpu_error}")
                            raise

                    # Re-raise if no fallback worked
                    raise
                else:
                    # Not an OOM error, re-raise
                    raise

        return wrapper

    return decorator


class AutoBatchProcessor:
    """Automatically adjust batch size to fit in GPU memory.

    Starts with a specified batch size and automatically reduces it
    if OOM errors occur, finding the optimal batch size for available memory.
    """

    def __init__(
        self,
        initial_batch_size: int = 32,
        min_batch_size: int = 1,
        reduction_factor: float = 0.5,
    ):
        """Initialize auto-batch processor.

        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum batch size before giving up
            reduction_factor: Factor to reduce batch size on OOM (0.5 = halve)
        """
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.reduction_factor = reduction_factor
        self.current_batch_size = initial_batch_size

    def process_with_auto_batch(
        self,
        data: list,
        process_fn: Callable,
        device: torch.device,
    ) -> list:
        """Process data with automatic batch size adjustment.

        Args:
            data: List of data items to process
            process_fn: Function that processes a batch (takes list, returns list)
            device: Device to use for processing

        Returns:
            List of processed results
        """
        results = []
        batch_size = self.current_batch_size

        i = 0
        while i < len(data):
            batch = data[i : i + batch_size]

            try:
                batch_results = process_fn(batch, device)
                results.extend(batch_results)
                i += batch_size

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Reduce batch size
                    new_batch_size = max(
                        self.min_batch_size, int(batch_size * self.reduction_factor)
                    )

                    if new_batch_size == batch_size:
                        # Can't reduce further
                        logger.error(
                            f"OOM with minimum batch size {batch_size}. "
                            "Cannot process this data."
                        )
                        raise

                    logger.warning(
                        f"OOM with batch size {batch_size}. "
                        f"Reducing to {new_batch_size}"
                    )
                    batch_size = new_batch_size
                    self.current_batch_size = new_batch_size

                    # Clear cache and retry
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise

        return results
