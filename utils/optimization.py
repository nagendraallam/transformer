"""
Optimization utilities for training transformer models.
"""

import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
import math
from typing import Optional, List, Dict, Any

try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False


def create_optimizer(
    model: torch.nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create an optimizer for training.
    
    Args:
        model: PyTorch model
        optimizer_name: Name of the optimizer (adamw, adam, sgd, lion)
        learning_rate: Learning rate
        weight_decay: Weight decay
        betas: Beta parameters for Adam-based optimizers
        eps: Epsilon for numerical stability
        **kwargs: Additional arguments for specific optimizers
        
    Returns:
        PyTorch optimizer
    """
    # Prepare optimizer parameters groups
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln_1.weight", "ln_2.weight"]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # Create optimizer based on name
    if optimizer_name.lower() == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            **kwargs
        )
    elif optimizer_name.lower() == "adam":
        optimizer = Adam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            **kwargs
        )
    elif optimizer_name.lower() == "sgd":
        optimizer = SGD(
            optimizer_grouped_parameters,
            lr=learning_rate,
            momentum=kwargs.get("momentum", 0.9),
            **kwargs
        )
    elif optimizer_name.lower() == "lion":
        if not LION_AVAILABLE:
            raise ImportError("Lion optimizer is not available. Install it with 'pip install lion-pytorch'")
        
        # Lion typically uses lower learning rates (1e-4 recommended)
        lion_lr = learning_rate * 0.1 if learning_rate > 1e-4 else learning_rate
        optimizer = Lion(
            optimizer_grouped_parameters,
            lr=lion_lr,
            betas=kwargs.get("lion_betas", (0.9, 0.99)),
            weight_decay=weight_decay,
            **{k: v for k, v in kwargs.items() if k != "lion_betas"}
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def get_linear_warmup_with_cosine_decay_schedule(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as a ratio of the initial learning rate
        last_epoch: Last epoch
        
    Returns:
        PyTorch learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_warmup_with_linear_decay_schedule(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a scheduler with linear warmup and linear decay.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as a ratio of the initial learning rate
        last_epoch: Last epoch
        
    Returns:
        PyTorch learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Linear decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 1.0 - progress)
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_after_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a scheduler with linear warmup and then constant learning rate.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        last_epoch: Last epoch
        
    Returns:
        PyTorch learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Constant
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "linear_warmup",
    num_warmup_steps: int = 0,
    num_training_steps: Optional[int] = None,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1,
    **kwargs
) -> Optional[LambdaLR]:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_name: Name of the scheduler
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as a ratio of the initial learning rate
        last_epoch: Last epoch
        **kwargs: Additional arguments for specific schedulers
        
    Returns:
        PyTorch learning rate scheduler or None if no scheduler is requested
    """
    if scheduler_name is None or scheduler_name.lower() == "none":
        return None
    
    if scheduler_name.lower() == "linear_warmup":
        if num_training_steps is None:
            return get_constant_after_warmup_schedule(
                optimizer, num_warmup_steps, last_epoch
            )
        else:
            return get_linear_warmup_with_linear_decay_schedule(
                optimizer, num_warmup_steps, num_training_steps, min_lr_ratio, last_epoch
            )
    elif scheduler_name.lower() == "cosine":
        if num_training_steps is None:
            raise ValueError("num_training_steps must be provided for cosine scheduler")
        
        return get_linear_warmup_with_cosine_decay_schedule(
            optimizer, num_warmup_steps, num_training_steps, min_lr_ratio, last_epoch
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}") 