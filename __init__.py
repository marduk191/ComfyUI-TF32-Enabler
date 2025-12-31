"""
ComfyUI TF32 Enabler
Automatically enables TensorFloat-32 acceleration for RTX 30/40/50 series GPUs

Author: marduk191
License: MIT
"""

from .enable_tf32 import enable_tf32, check_tf32_status

# Enable TF32 immediately on ComfyUI startup
enable_tf32(verbose=True)

# Required for ComfyUI custom node registration
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
