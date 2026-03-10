"""
TF32 Enabler for ComfyUI Custom Nodes
Optimizes performance on RTX 30/40/50 series GPUs
Includes torch.compile CUDA allocator fix
"""

import torch
import os


def fix_cuda_allocator():
    """
    Fix CUDA allocator issues with torch.compile.
    
    Resolves: "cudaMallocAsync does not yet support checkPoolLiveAllocations"
    """
    # Set expandable segments for better memory management with torch.compile
    current_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
    
    if 'expandable_segments' not in current_conf:
        if current_conf:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f"{current_conf},expandable_segments:True"
        else:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"


def enable_tf32(verbose=True):
    """
    Enable TensorFloat-32 for CUDA operations.
    
    TF32 provides ~1.5-2x speedup on Ampere/Ada/Blackwell GPUs
    with minimal precision impact for diffusion models.
    
    Args:
        verbose (bool): Print status message
    
    Returns:
        bool: True if enabled successfully, False otherwise
    """
    try:
        # Fix CUDA allocator for torch.compile compatibility
        fix_cuda_allocator()
        
        # Enable TF32 for matrix multiplications
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Enable TF32 for cuDNN operations
        torch.backends.cudnn.allow_tf32 = True
        
        if verbose:
            print("\n" + "=" * 60)
            print("🚀 ComfyUI TF32 Acceleration Enabled")
            print("=" * 60)
            print(f"   Matmul TF32: {torch.backends.cuda.matmul.allow_tf32}")
            print(f"   cuDNN TF32:  {torch.backends.cudnn.allow_tf32}")
            print(f"   CUDA Allocator: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')}")
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                compute_cap = torch.cuda.get_device_capability(0)
                print(f"   GPU: {device_name}")
                print(f"   Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
                
                # Warn if GPU doesn't support TF32
                if compute_cap[0] < 8:
                    print("   ⚠️  Note: TF32 requires Ampere or newer (compute capability >= 8.0)")
            else:
                print("   ⚠️  CUDA not available")
            print("   ✅ torch.compile CUDA allocator fix applied")

            try:
                torch.set_float32_matmul_precision("high")  # "high" or "medium"
                print(f"   ✅  float32 matmul precision = {torch.get_float32_matmul_precision()}\n")
            except Exception as e:
                print(f"   ⚠️   failed to set float32 matmul precision: {e}\n")

            print("=" * 60 + "\n")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"\n⚠️  Failed to enable TF32: {e}\n")
        return False


def check_tf32_status():
    """
    Check current TF32 status.
    
    Returns:
        dict: Status of TF32 settings
    """
    status = {
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        status["device_name"] = torch.cuda.get_device_name(0)
        status["compute_capability"] = torch.cuda.get_device_capability(0)
    
    return status
