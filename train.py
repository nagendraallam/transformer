#!/usr/bin/env python
"""
Training script for the transformer model.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import random_split
import numpy as np
import random
from tqdm import tqdm

from model import Transformer
from utils import (
    get_tokenizer,
    prepare_dummy_data,
    get_dataloaders,
    TransformerTrainer,
    create_scheduler,
)
from utils.optimization import create_optimizer
from utils.distributed import run_distributed, get_device, get_data_sampler, is_main_process
from configs.model_config import (
    SMALL_CONFIG, 
    MEDIUM_CONFIG, 
    LARGE_CONFIG, 
    TINY_CONFIG, 
    TRAINING_CONFIG, 
    OPTIMIZATION_CONFIG
)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_worker(
    args: argparse.Namespace,
    local_rank: int = 0,
    world_size: int = 1,
):
    """
    Worker function for training.
    
    Args:
        args: Command line arguments
        local_rank: Local rank of the process
        world_size: Number of processes in total
    """
    is_main = is_main_process(local_rank)
    
    # Get device
    if args.device is None:
        device = get_device(local_rank) if args.distributed else "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    if is_main:
        print(f"Using device: {device}")
    
    # Check if mixed precision is compatible with device
    if args.mixed_precision and "cuda" not in str(device):
        if is_main:
            print("Warning: Mixed precision training is only supported on CUDA devices. It will be automatically disabled.")
        args.mixed_precision = False
    
    # Set random seed
    set_seed(args.seed + local_rank)  # Add rank to make seeds different across ranks
    
    # Select model configuration
    if args.model_size.lower() == "small":
        model_config = SMALL_CONFIG.copy()
    elif args.model_size.lower() == "medium":
        model_config = MEDIUM_CONFIG.copy()
    elif args.model_size.lower() == "large":
        model_config = LARGE_CONFIG.copy()
    elif args.model_size.lower() == "tiny":
        model_config = TINY_CONFIG.copy()
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
    
    # Update model configuration with command line arguments
    model_config["max_seq_length"] = args.max_seq_length
    
    # Override use_rotary_embeddings if explicitly specified
    if args.use_rotary_embeddings:
        model_config["use_rotary_embeddings"] = True
    
    # Override activation if explicitly specified
    if args.activation != "gelu":
        model_config["activation"] = args.activation
    
    # Prepare optimization configuration
    opt_config = OPTIMIZATION_CONFIG.copy()
    opt_config["mixed_precision"] = args.mixed_precision
    opt_config["gradient_checkpointing"] = args.gradient_checkpointing
    opt_config["bf16"] = args.bf16
    opt_config["use_flash_attention"] = args.flash_attention
    
    # Initialize tokenizer
    tokenizer = get_tokenizer(max_length=args.max_seq_length)
    
    # Prepare data
    if args.dummy_data:
        if is_main:
            print("Using dummy data for training")
        dataset = prepare_dummy_data(
            tokenizer=tokenizer,
            num_samples=args.dummy_samples,
            seq_length=args.dummy_seq_length,
        )
    elif args.data is not None:
        # Load data based on file extension
        file_ext = os.path.splitext(args.data)[1].lower()
        
        if file_ext == ".json":
            from utils import load_json_dataset
            dataset = load_json_dataset(
                args.data, 
                tokenizer, 
                args.max_seq_length, 
                cache_tokenization=True
            )
        elif file_ext == ".txt":
            from utils import load_text_file
            dataset = load_text_file(
                args.data, 
                tokenizer, 
                args.max_seq_length, 
                cache_tokenization=True,
                streaming=args.streaming_data
            )
        elif file_ext in [".csv", ".tsv"]:
            from utils import load_csv_dataset
            dataset = load_csv_dataset(
                args.data, 
                tokenizer, 
                args.max_seq_length, 
                cache_tokenization=True
            )
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
    else:
        raise ValueError("Either --data or --dummy_data must be specified")
    
    # Split data into train and validation sets
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    # Use different random seeds for different ranks for a better split in distributed training
    generator = torch.Generator().manual_seed(args.seed + local_rank)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    if is_main:
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
    
    # Get sampling strategy for distributed training
    train_sampler = None
    if args.distributed:
        train_sampler = get_data_sampler(train_dataset, shuffle=True, distributed=True)
        val_sampler = get_data_sampler(val_dataset, shuffle=False, distributed=True)
    
    # Create dataloaders
    dataloaders = get_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler,
        pin_memory=True,
        use_bucketing=args.bucketing,
    )
    
    # Initialize model
    model = Transformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        num_encoder_layers=model_config["num_encoder_layers"],
        num_decoder_layers=model_config["num_decoder_layers"],
        d_ff=model_config["d_ff"],
        max_seq_length=model_config["max_seq_length"],
        dropout=model_config["dropout"],
        activation=model_config.get("activation", "gelu"),
        use_rotary_embeddings=model_config.get("use_rotary_embeddings", True),
    )
    
    # Print model parameter count
    if is_main:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {param_count:,} trainable parameters")
    
    # Calculate total training steps
    total_steps = len(dataloaders["train"]) * args.epochs // args.gradient_accumulation_steps
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model=model,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_name=args.scheduler,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Initialize trainer with optimization options
    trainer = TransformerTrainer(
        model=model,
        train_dataloader=dataloaders["train"],
        tokenizer=tokenizer,
        val_dataloader=dataloaders["val"],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        experiment_dir=args.save_dir,
        log_interval=args.log_interval,
        mixed_precision=opt_config["mixed_precision"],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=opt_config["gradient_checkpointing"],
        bf16=opt_config["bf16"],
        save_best_only=args.save_best_only,
        use_distributed=args.distributed,
        local_rank=local_rank,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        use_flash_attention=opt_config["use_flash_attention"],
    )
    
    # Load checkpoint if specified
    if args.resume_from:
        if is_main:
            print(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # Train the model
    if is_main:
        print(f"Starting training for {args.epochs} epochs")
        if opt_config["mixed_precision"] and "cuda" in str(device):
            print(f"Using: Mixed precision training ({('bfloat16' if opt_config['bf16'] else 'float16')})")
        else:
            print("Using: Full precision training")
        
        if opt_config["gradient_checkpointing"]:
            print("Using gradient checkpointing for memory efficiency")
        
        if args.compile:
            print(f"Using torch.compile with mode: {args.compile_mode}")
            
        if opt_config["use_flash_attention"]:
            print("Using flash attention for faster computation")
    
    training_history = trainer.train(num_epochs=args.epochs)
    
    # Save outputs only on main process
    if is_main:
        # Plot training history
        trainer.plot_training_history(save_path=os.path.join(args.save_dir, "training_history.png"))
        
        # Save final model
        final_model_path = os.path.join(args.save_dir, "final_model.pt")
        trainer.save_checkpoint(final_model_path)
        
        # Save model weights only (smaller file size)
        weights_only_path = os.path.join(args.save_dir, "model_weights_only.pt")
        torch.save(model.state_dict(), weights_only_path)
        print(f"Model weights saved to {weights_only_path}")
        
        print("Training completed!")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"Best validation perplexity: {trainer.perplexity(trainer.best_val_loss):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train a transformer model")
    
    # Data parameters
    parser.add_argument("--data", type=str, default=None, help="Path to data file")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--streaming_data", action="store_true", help="Use streaming data for large datasets")
    
    # Model parameters
    parser.add_argument("--model_size", type=str, default="small", help="Model size: small, medium, large")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--use_rotary_embeddings", action="store_true", help="Use rotary embeddings")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function (gelu, relu)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps for scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer: adamw, adam, adafactor, lion")
    parser.add_argument("--scheduler", type=str, default="linear_warmup", help="Scheduler: linear_warmup, cosine_warmup, step")
    
    # Optimization parameters
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision instead of float16")
    parser.add_argument("--flash_attention", action="store_true", help="Use flash attention")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for faster training")
    parser.add_argument("--compile_mode", type=str, default="default", help="Mode for torch.compile: default, reduce-overhead, max-autotune")
    
    # Data loading parameters
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--bucketing", action="store_true", help="Use bucketing for efficient batching")
    
    # Distributed training parameters
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--world_size", type=int, default=None, help="Number of processes for distributed training")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu)")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_best_only", action="store_true", help="Only save the best model")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--dummy_data", action="store_true", help="Use dummy data for testing")
    parser.add_argument("--dummy_samples", type=int, default=1000, help="Number of dummy samples")
    parser.add_argument("--dummy_seq_length", type=int, default=64, help="Sequence length for dummy data")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume training from a checkpoint")
    
    args = parser.parse_args()
    
    # Set world_size if not specified
    if args.distributed and args.world_size is None:
        if torch.cuda.is_available():
            args.world_size = torch.cuda.device_count()
        else:
            args.world_size = 1
    
    # Run distributed or single process training
    if args.distributed and args.world_size > 1:
        run_distributed(train_worker, args.world_size, args=args)
    else:
        train_worker(args)


if __name__ == "__main__":
    main() 