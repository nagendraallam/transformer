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
    create_optimizer,
    create_scheduler,
)
from configs.model_config import (
    SMALL_CONFIG, 
    MEDIUM_CONFIG, 
    LARGE_CONFIG, 
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


def main():
    parser = argparse.ArgumentParser(description="Train a transformer model")
    
    # Data parameters
    parser.add_argument("--data", type=str, default=None, help="Path to data file")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    
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
    
    # Optimization parameters
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision instead of float16")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu)")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_best_only", action="store_true", help="Only save the best model")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--dummy_data", action="store_true", help="Use dummy data for testing")
    parser.add_argument("--dummy_samples", type=int, default=1000, help="Number of dummy samples")
    parser.add_argument("--dummy_seq_length", type=int, default=64, help="Sequence length for dummy data")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Select model configuration
    if args.model_size.lower() == "small":
        model_config = SMALL_CONFIG.copy()
    elif args.model_size.lower() == "medium":
        model_config = MEDIUM_CONFIG.copy()
    elif args.model_size.lower() == "large":
        model_config = LARGE_CONFIG.copy()
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
    
    # Initialize tokenizer
    tokenizer = get_tokenizer(max_length=args.max_seq_length)
    
    # Prepare data
    if args.dummy_data:
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
            dataset = load_json_dataset(args.data, tokenizer, args.max_seq_length)
        elif file_ext == ".txt":
            from utils import load_text_file
            dataset = load_text_file(args.data, tokenizer, args.max_seq_length)
        elif file_ext in [".csv", ".tsv"]:
            from utils import load_csv_dataset
            dataset = load_csv_dataset(args.data, tokenizer, args.max_seq_length)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
    else:
        raise ValueError("Either --data or --dummy_data must be specified")
    
    # Split data into train and validation sets
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    dataloaders = get_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
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
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {param_count:,} trainable parameters")
    
    # Calculate total training steps
    total_steps = len(dataloaders["train"]) * args.epochs // args.gradient_accumulation_steps
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model=model,
        optimizer_name="adamw",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_name="linear_warmup",
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Initialize trainer with optimization options
    trainer = TransformerTrainer(
        model=model,
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders["val"],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.save_dir,
        log_interval=args.log_interval,
        mixed_precision=opt_config["mixed_precision"],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=opt_config["gradient_checkpointing"],
        bf16=opt_config["bf16"],
        save_best_only=args.save_best_only,
    )
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs")
    print(f"Using: {'Mixed precision' if opt_config['mixed_precision'] else 'Full precision'} training")
    if opt_config["gradient_checkpointing"]:
        print("Using gradient checkpointing for memory efficiency")
    
    training_history = trainer.train(num_epochs=args.epochs)
    
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


if __name__ == "__main__":
    main() 