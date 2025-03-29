#!/usr/bin/env python
"""
Fine-tuning script for pre-trained GPT-2 models.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import numpy as np
import random

from model import PreTrainedGPT2
from utils import TransformerTrainer, TextDataset, load_text_file, get_dataloaders
from utils.tokenizer import get_tokenizer
from utils.optimization import create_optimizer, create_scheduler
from utils.data_utils import prepare_dummy_data


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained GPT-2 model")
    
    # Data arguments
    parser.add_argument("--data", type=str, help="Path to training data file")
    parser.add_argument("--dummy_data", action="store_true", help="Use dummy data for testing")
    parser.add_argument("--dummy_samples", type=int, default=100, help="Number of dummy samples")
    parser.add_argument("--dummy_seq_length", type=int, default=64, help="Length of dummy sequences")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    
    # Model arguments
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "medium", "large", "xl"],
                       help="GPT-2 model size (small=base gpt2, medium, large, xl)")
    parser.add_argument("--fine_tune_all", action="store_true", help="Fine-tune all parameters")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=200, help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--save_best_only", action="store_true", help="Save only the best model")
    
    # Mixed precision arguments
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu)")

    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Build optimization config
    opt_config = {
        "mixed_precision": args.mixed_precision,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    
    # Set model name based on size
    model_name_mapping = {
        "small": "gpt2",
        "medium": "gpt2-medium",
        "large": "gpt2-large",
        "xl": "gpt2-xl"
    }
    pretrained_model_name = model_name_mapping[args.model_size]
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = get_tokenizer(pretrained_model_name, max_length=args.max_seq_length)
    
    # Prepare dataset
    if args.dummy_data:
        print("Using dummy data for training")
        dataset = prepare_dummy_data(
            tokenizer=tokenizer,
            num_samples=args.dummy_samples,
            seq_length=args.dummy_seq_length,
        )
        train_size = int(len(dataset) * 0.9)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    else:
        if not args.data:
            raise ValueError("Either --data or --dummy_data must be specified")
        
        print(f"Loading data from {args.data}")
        dataset = load_text_file(
            file_path=args.data,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
        )
        
        # Split into train and validation sets
        train_size = int(len(dataset) * (1 - args.val_split))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    dataloaders = get_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Initialize model
    model = PreTrainedGPT2(
        pretrained_model_name=pretrained_model_name,
        vocab_size=tokenizer.vocab_size,
        fine_tune=args.fine_tune_all,
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
    )
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs")
    if opt_config["mixed_precision"] and device.startswith("cuda"):
        print(f"Using: Mixed precision training ({('bfloat16' if opt_config['bf16'] else 'float16')})")
    else:
        print("Using: Full precision training")
    
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