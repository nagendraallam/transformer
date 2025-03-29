"""
Example script that demonstrates how to train the transformer model on custom data.
"""

import os
import sys
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import Transformer
from utils import (
    get_tokenizer,
    load_text_file,
    get_dataloaders,
    TransformerTrainer,
    create_optimizer,
    create_scheduler,
)
from configs.model_config import SMALL_CONFIG


def train_on_custom_data(
    data_file: str,
    model_size: str = "small",
    batch_size: int = 4,
    epochs: int = 5,
    learning_rate: float = 5e-5,
    max_seq_length: int = 512,
    val_split: float = 0.1,
    mixed_precision: bool = False,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    bf16: bool = False,
    save_best_only: bool = True,
):
    """
    Train the transformer model on custom data.
    
    Args:
        data_file: Path to the data file (text file with one example per line)
        model_size: Size of the model ("small", "medium", "large")
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate
        max_seq_length: Maximum sequence length
        val_split: Validation split ratio
        mixed_precision: Whether to use mixed precision training
        gradient_accumulation_steps: Number of steps to accumulate gradients
        gradient_checkpointing: Whether to use gradient checkpointing to save memory
        bf16: Whether to use bfloat16 precision instead of float16
        save_best_only: Whether to save only the best model
    """
    # Select configuration based on model size
    if model_size.lower() == "small":
        from configs.model_config import SMALL_CONFIG as model_config
    elif model_size.lower() == "medium":
        from configs.model_config import MEDIUM_CONFIG as model_config
    elif model_size.lower() == "large":
        from configs.model_config import LARGE_CONFIG as model_config
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    # Update max sequence length
    model_config["max_seq_length"] = max_seq_length
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = get_tokenizer(max_length=max_seq_length)
    
    # Load data
    print(f"Loading data from {data_file}")
    dataset = load_text_file(data_file, tokenizer, max_seq_length)
    
    # Create train and validation splits
    from torch.utils.data import random_split
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    if val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    else:
        train_dataset, val_dataset = dataset, None
    
    print(f"Train dataset size: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    dataloaders = get_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
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
    )
    
    # Calculate total training steps
    total_steps = len(dataloaders["train"]) * epochs // gradient_accumulation_steps
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model=model,
        optimizer_name="adamw",
        learning_rate=learning_rate,
        weight_decay=0.01,
    )
    
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_name="linear_warmup",
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=total_steps,
    )
    
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Initialize trainer
    trainer = TransformerTrainer(
        model=model,
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders.get("val"),
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir="checkpoints",
        log_interval=10,
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        bf16=bf16,
        save_best_only=save_best_only,
    )
    
    # Train model
    print(f"Starting training for {epochs} epochs")
    if mixed_precision:
        print(f"Using {'bfloat16' if bf16 else 'float16'} mixed precision training")
    if gradient_checkpointing:
        print("Using gradient checkpointing for memory efficiency")
    
    trainer.train(num_epochs=epochs)
    
    # Save final model
    final_model_path = os.path.join("checkpoints", "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model_config,
    }, final_model_path)
    
    print(f"Training completed. Final model saved to {final_model_path}")
    return final_model_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train transformer on custom data")
    
    parser.add_argument("--data_file", type=str, required=True, help="Path to data file (text file with one example per line)")
    parser.add_argument("--model_size", type=str, default="small", help="Model size: small, medium, large")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision (instead of float16)")
    parser.add_argument("--save_best_only", action="store_true", help="Save only the best model")
    
    args = parser.parse_args()
    
    train_on_custom_data(
        data_file=args.data_file,
        model_size=args.model_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        val_split=args.val_split,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        save_best_only=args.save_best_only,
    ) 