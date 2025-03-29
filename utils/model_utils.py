import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Any

def save_model_weights_only(
    model: nn.Module,
    output_path: str,
    half_precision: bool = False,
    bf16_precision: bool = False,
) -> str:
    """
    Save only the model weights (state_dict) without optimizer state and other training info.
    
    Args:
        model: The model to save
        output_path: Path to save the model weights
        half_precision: Whether to save in float16 precision
        bf16_precision: Whether to save in bfloat16 precision

    Returns:
        Path to the saved model weights
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Clone the model to avoid modifying the original
    model_copy = type(model)(*model.__init_args__, **model.__init_kwargs__)
    model_copy.load_state_dict(model.state_dict())
    
    # Convert to half precision if requested
    if half_precision:
        model_copy = model_copy.half()
        if not output_path.endswith(".pt"):
            output_path = f"{os.path.splitext(output_path)[0]}_fp16.pt"
        else:
            output_path = f"{os.path.splitext(output_path)[0]}_fp16.pt"
    
    # Convert to bf16 precision if requested
    if bf16_precision:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            model_copy = model_copy.to(torch.bfloat16)
            if not output_path.endswith(".pt"):
                output_path = f"{os.path.splitext(output_path)[0]}_bf16.pt"
            else:
                output_path = f"{os.path.splitext(output_path)[0]}_bf16.pt"
        else:
            print("BF16 precision not supported on this device. Saving in original precision.")
    
    # Save only the model weights
    torch.save(model_copy.state_dict(), output_path)
    
    # Print size of the saved file
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved model weights to {output_path} ({file_size_mb:.2f} MB)")
    
    return output_path


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    step: int,
    loss: float,
    output_path: str,
    additional_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a complete checkpoint including model weights, optimizer state, scheduler, etc.
    
    Args:
        model: The model to save
        optimizer: The optimizer
        scheduler: The learning rate scheduler
        epoch: Current epoch number
        step: Current step number
        loss: Current loss value
        output_path: Path to save the checkpoint
        additional_info: Any additional information to save in the checkpoint

    Returns:
        Path to the saved checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss,
    }
    
    # Add optimizer state
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Add scheduler state
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Add additional info
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    # Save checkpoint
    torch.save(checkpoint, output_path)
    
    # Print size of the saved file
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved checkpoint to {output_path} ({file_size_mb:.2f} MB)")
    
    return output_path


def load_model_weights(
    model: nn.Module,
    weights_path: str,
    device: str = 'cpu',
    strict: bool = True,
) -> nn.Module:
    """
    Load model weights from a saved state dict.
    
    Args:
        model: The model to load weights into
        weights_path: Path to the saved weights
        device: Device to load the weights to
        strict: Whether to strictly enforce that the keys in state_dict match the keys in model
        
    Returns:
        Model with loaded weights
    """
    # Load state dict
    state_dict = torch.load(weights_path, map_location=device)
    
    # If the saved file is a complete checkpoint rather than just weights
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    # Load weights into model
    model.load_state_dict(state_dict, strict=strict)
    
    return model


def get_model_size(model: nn.Module) -> Dict[str, int]:
    """
    Calculate the number of parameters in a model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
    }


def print_model_size(model: nn.Module) -> None:
    """
    Print formatted information about model size.
    
    Args:
        model: The model to analyze
    """
    size_info = get_model_size(model)
    
    print(f"Model Size Information:")
    print(f"  Total parameters: {size_info['total_params']:,}")
    print(f"  Trainable parameters: {size_info['trainable_params']:,}")
    print(f"  Non-trainable parameters: {size_info['non_trainable_params']:,}")
    
    # Estimate memory usage
    params_mb = size_info['total_params'] * 4 / (1024 * 1024)  # 4 bytes for float32
    
    print(f"  Estimated memory for parameters: {params_mb:.2f} MB (FP32)")
    print(f"  Estimated memory for parameters: {params_mb/2:.2f} MB (FP16)")
    
    if torch.cuda.is_available():
        # Try to estimate CUDA memory usage during training
        # Very rough estimate: ~3x params for optimizer states and gradients
        cuda_mem = params_mb * 3
        print(f"  Estimated CUDA memory during training: ~{cuda_mem:.2f} MB (FP32)") 