import importlib
import importlib.metadata as imp_meta
import torch
import logging
import argparse

_logger = logging.getLogger(__name__)


def is_torch_npu_available():
    _torch_npu_available = importlib.util.find_spec("torch_npu") is not None
    if _torch_npu_available:
        try:
            torch_npu_version = imp_meta.version("torch_npu")
            import torch_npu  # noqa: F401
            torch.npu.set_device(0)
            _logger.info(f"torch_npu version {torch_npu_version} is available.")
        except ImportError:
            _torch_npu_available = False
    return _torch_npu_available

is_torch_npu_available()

