"""
Distributed training utilities for the transformer model.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Dict, Any, Callable


def setup_distributed(rank: int, world_size: int) -> None:
    """
    Initialize the distributed environment.
    
    Args:
        rank: Rank of the current process
        world_size: Number of processes in the group
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", 
                            rank=rank, 
                            world_size=world_size)


def cleanup_distributed() -> None:
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device(local_rank: int) -> torch.device:
    """
    Get the device for the current process.
    
    Args:
        local_rank: Local rank of the process
        
    Returns:
        Device for the current process
    """
    if torch.cuda.is_available():
        return torch.device(f'cuda:{local_rank}')
    return torch.device('cpu')


def wrap_model_for_distributed(model: torch.nn.Module, device: torch.device) -> DDP:
    """
    Wrap a model for distributed training.
    
    Args:
        model: PyTorch model to wrap
        device: Device to use
        
    Returns:
        Wrapped model for distributed training
    """
    # Move model to the assigned device
    model.to(device)
    
    # Wrap the model with DistributedDataParallel
    ddp_model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)
    
    return ddp_model


def run_distributed(
    fn: Callable,
    world_size: int,
    *args: Any,
    **kwargs: Any
) -> None:
    """
    Run a function in a distributed setting.
    
    Args:
        fn: Function to run
        world_size: Number of processes
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
    """
    mp.spawn(
        _distributed_worker,
        args=(fn, world_size, args, kwargs),
        nprocs=world_size,
        join=True
    )


def _distributed_worker(
    rank: int,
    fn: Callable,
    world_size: int,
    args: tuple,
    kwargs: dict
) -> None:
    """
    Worker function for distributed training.
    
    Args:
        rank: Rank of the current process
        fn: Function to run
        world_size: Number of processes
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
    """
    setup_distributed(rank, world_size)
    
    # Add rank and world_size to kwargs
    kwargs['rank'] = rank
    kwargs['world_size'] = world_size
    kwargs['local_rank'] = rank
    
    # Call the function
    try:
        fn(*args, **kwargs)
    finally:
        cleanup_distributed()


def is_main_process(local_rank: int) -> bool:
    """
    Check if the current process is the main process.
    
    Args:
        local_rank: Local rank of the process
        
    Returns:
        True if the current process is the main process, False otherwise
    """
    return local_rank == 0


def get_data_sampler(dataset, shuffle: bool = True, distributed: bool = False):
    """
    Get a data sampler for distributed training.
    
    Args:
        dataset: PyTorch dataset
        shuffle: Whether to shuffle the dataset
        distributed: Whether to use distributed training
        
    Returns:
        Appropriate sampler for the dataset
    """
    if distributed:
        # Use DistributedSampler for distributed training
        from torch.utils.data import DistributedSampler
        return DistributedSampler(dataset, shuffle=shuffle)
    
    if shuffle:
        # Use RandomSampler for shuffling in non-distributed setting
        from torch.utils.data import RandomSampler
        return RandomSampler(dataset)
    
    # Use SequentialSampler for non-shuffling in non-distributed setting
    from torch.utils.data import SequentialSampler
    return SequentialSampler(dataset) 