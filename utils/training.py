"""
Training utilities for the transformer model.
"""

import os
import time
import math
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from contextlib import nullcontext


class TransformerTrainer:
    """Trainer for transformer models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        tokenizer,  # Tokenizer object for decoding predictions
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        val_dataloader: Optional[DataLoader] = None,
        device: str = "cpu",
        experiment_dir: str = "./experiments",
        gradient_accumulation_steps: int = 1,
        log_interval: int = 10,
        gradient_checkpointing: bool = False,
        mixed_precision: bool = False,
        bf16: bool = False,
        save_best_only: bool = False,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            train_dataloader: DataLoader for training data
            tokenizer: Tokenizer for decoding predictions
            criterion: Loss function (defaults to CrossEntropyLoss)
            optimizer: Optimizer (defaults to AdamW)
            scheduler: Learning rate scheduler
            val_dataloader: DataLoader for validation data
            device: Device to train on
            experiment_dir: Directory to save checkpoints and logs
            gradient_accumulation_steps: Number of steps to accumulate gradients before updating weights
            log_interval: Log training metrics every N steps
            gradient_checkpointing: Whether to use gradient checkpointing to save memory
            mixed_precision: Whether to use mixed precision training
            bf16: Whether to use bfloat16 precision instead of float16
            save_best_only: Whether to save only the best model during training
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        self.device = device
        self.experiment_dir = experiment_dir
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_interval = log_interval
        self.gradient_checkpointing = gradient_checkpointing
        self.bf16 = bf16
        self.save_best_only = save_best_only
        
        # Move model to the specified device
        self.model.to(self.device)
        
        # Set up criterion if not provided
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        
        # Set up optimizer if not provided
        self.optimizer = optimizer if optimizer is not None else torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        # Set up scheduler
        self.scheduler = scheduler
        
        # Create experiment directory if it doesn't exist
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize mixed precision settings
        self.mixed_precision = mixed_precision
        self.scaler = None  # Initialize scaler to None
        self.amp_dtype = None
        
        if self.mixed_precision:
            # Check if we're on a CUDA device
            if self.device.startswith("cuda"):
                # Use proper torch.amp instead of torch.cuda.amp which is deprecated
                if self.bf16 and torch.cuda.is_bf16_supported():
                    self.amp_dtype = torch.bfloat16
                    print("Using bfloat16 precision with native type conversion")
                else:
                    self.amp_dtype = torch.float16
                    # Initialize gradient scaler only with float16 (not needed for bf16)
                    self.scaler = torch.amp.GradScaler()
                    print("Using float16 precision with gradient scaling")
            else:
                print("Warning: Mixed precision training is only supported on CUDA devices. Using full precision.")
                self.mixed_precision = False
        
        # Setup gradient checkpointing if requested
        if self.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        
        # Initialize tracking variables
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.training_history = {"train_loss": [], "val_loss": [], "learning_rate": []}
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the loss for a batch.
        
        Args:
            batch: Batch of data from DataLoader
            
        Returns:
            Loss value
        """
        # Check if we're doing sequence-to-sequence training
        if "decoder_input_ids" in batch:
            # Seq2Seq transformer model forward pass
            outputs = self.model(
                src=batch["input_ids"].to(self.device),
                tgt=batch["decoder_input_ids"][:, :-1].to(self.device),  # Remove last token (as we predict next)
                src_mask=batch["attention_mask"].to(self.device),
                tgt_mask=None,  # Will be created automatically in the model
            )
            
            # Flatten outputs for loss computation
            outputs = outputs.contiguous().view(-1, outputs.size(-1))
            targets = batch["labels"][:, 1:].contiguous().view(-1).to(self.device)  # Remove first token (BOS)
            
        else:
            # Language model style forward pass
            outputs = self.model(
                src=batch["input_ids"].to(self.device),
                tgt=batch["input_ids"].to(self.device),
                src_mask=batch["attention_mask"].to(self.device),
            )
            
            # Flatten outputs for loss computation
            outputs = outputs[:, :-1].contiguous().view(-1, outputs.size(-1))
            targets = batch["input_ids"][:, 1:].contiguous().view(-1).to(self.device)
        
        # Compute loss
        loss = self.criterion(outputs, targets)
        return loss
    
    def train_epoch(self) -> float:
        """
        Train the model for one epoch.
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Use mixed precision for forward pass if enabled
            if self.mixed_precision:
                with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype) if self.device.startswith('cuda') else nullcontext():
                    loss = self._compute_loss(batch)
                
                # Scale the loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                # Use scaler for backward pass if using float16
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                loss = self._compute_loss(batch)
                # Scale the loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            # Only update parameters after accumulating gradients
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                if self.scaler is not None:
                    # Unscale gradients for clipping with float16
                    self.scaler.unscale_(self.optimizer)
                    
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                if self.scaler is not None:
                    # Update with scaler for float16
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Regular update for fp32 or bf16
                    self.optimizer.step()
                
                # Step the learning rate scheduler if it exists
                if self.scheduler is not None:
                    self.scheduler.step()
                    self.training_history["learning_rate"].append(self.scheduler.get_last_lr()[0])
                
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
            
            # Accumulate unscaled loss for reporting
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            avg_loss = epoch_loss / (batch_idx + 1)
            
            # Update progress bar
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
            
            # Log at intervals
            if (batch_idx + 1) % self.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Step {self.global_step} | Loss: {avg_loss:.4f} | {elapsed:.2f}s elapsed")
        
        avg_epoch_loss = epoch_loss / len(self.train_dataloader)
        self.training_history["train_loss"].append(avg_epoch_loss)
        
        return avg_epoch_loss
    
    def validate(self) -> float:
        """
        Validate the model on the validation set.
        
        Returns:
            Average loss on the validation set
        """
        if self.val_dataloader is None:
            return 0.0
        
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                # Use mixed precision for evaluation if enabled
                if self.mixed_precision:
                    with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype) if self.device.startswith('cuda') else nullcontext():
                        loss = self._compute_loss(batch)
                else:
                    loss = self._compute_loss(batch)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(self.val_dataloader)
        self.training_history["val_loss"].append(avg_val_loss)
        
        return avg_val_loss
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train for
            
        Returns:
            Training history
        """
        print(f"Training for {num_epochs} epochs on {self.device}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M parameters")
        
        for epoch in range(self.current_epoch, self.current_epoch + num_epochs):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch + 1}/{self.current_epoch + num_epochs}")
            
            # Train for one epoch
            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate if we have a validation set
            if self.val_dataloader is not None:
                val_loss = self.validate()
                print(f"Validation Loss: {val_loss:.4f}")
                print(f"Validation Perplexity: {self.perplexity(val_loss):.2f}")
                
                # Save model if it's the best so far
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if self.save_best_only:
                        checkpoint_path = self.save_checkpoint(is_best=True)
                        print(f"Saved best model checkpoint to {checkpoint_path}")
                    else:
                        checkpoint_path = self.save_checkpoint(is_best=True)
                        print(f"Saved model checkpoint to {checkpoint_path}")
            
            # If no validation dataloader or save_best_only is False, save at each epoch
            if self.val_dataloader is None or not self.save_best_only:
                checkpoint_path = self.save_checkpoint(is_best=False)
                print(f"Saved model checkpoint to {checkpoint_path}")
        
        return self.training_history
    
    def save_checkpoint(self, is_best: bool = False) -> str:
        """
        Save a checkpoint of the model, optimizer, and training state.
        
        Args:
            is_best: Whether this checkpoint is the best model so far
            
        Returns:
            Path to the saved checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save the checkpoint
        if is_best:
            checkpoint_path = os.path.join(self.experiment_dir, "best_model.pt")
        else:
            checkpoint_path = os.path.join(self.experiment_dir, f"checkpoint_epoch_{self.current_epoch + 1}.pt")
        
        # Also save the latest checkpoint
        latest_path = os.path.join(self.experiment_dir, "latest_checkpoint.pt")
        
        # Save the checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Save a copy as the latest checkpoint
        torch.save(checkpoint, latest_path)
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint to load
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint["training_history"]
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch + 1}")
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot the training history.
        
        Args:
            save_path: Path to save the plot to (optional)
        """
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history["train_loss"], label="Train Loss")
        if self.training_history["val_loss"]:
            plt.plot(self.training_history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        
        # Plot learning rate
        if self.training_history["learning_rate"]:
            plt.subplot(1, 2, 2)
            plt.plot(self.training_history["learning_rate"])
            plt.xlabel("Step")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved training history plot to {save_path}")
        
        plt.close()
    
    def perplexity(self, loss: float) -> float:
        """
        Calculate perplexity from loss.
        
        Args:
            loss: Cross-entropy loss
            
        Returns:
            Perplexity
        """
        return math.exp(loss)


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create an optimizer for the model.
    
    Args:
        model: The model to optimize
        optimizer_name: Name of the optimizer to use
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 penalty)
        **kwargs: Additional arguments to pass to the optimizer
        
    Returns:
        Initialized optimizer
    """
    no_decay = ["bias", "LayerNorm.weight"]
    
    # Group parameters to apply different weight decay
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
    
    # Create optimizer
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == "adam":
        return optim.Adam(optimizer_grouped_parameters, lr=learning_rate, **kwargs)
    elif optimizer_name == "adamw":
        return optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, **kwargs)
    elif optimizer_name == "sgd":
        return optim.SGD(optimizer_grouped_parameters, lr=learning_rate, **kwargs)
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(optimizer_grouped_parameters, lr=learning_rate, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "linear_warmup",
    num_warmup_steps: int = 0,
    num_training_steps: int = 0,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: The optimizer to schedule
        scheduler_name: Name of the scheduler to use
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        **kwargs: Additional arguments to pass to the scheduler
        
    Returns:
        Initialized scheduler
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == "linear_warmup":
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **kwargs
        )
    elif scheduler_name == "cosine_warmup":
        from transformers import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **kwargs
        )
    elif scheduler_name == "step":
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}") 