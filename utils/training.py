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


class TransformerTrainer:
    """Trainer for transformer models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[Callable] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 100,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Transformer model
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data (optional)
            optimizer: Optimizer for training (optional, defaults to Adam)
            scheduler: Learning rate scheduler (optional)
            criterion: Loss function (optional, defaults to CrossEntropyLoss)
            device: Device to use for training ("cuda" or "cpu")
            checkpoint_dir: Directory to save checkpoints to
            log_interval: Number of steps between logging training status
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Default optimizer is Adam
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=1e-4)
        self.scheduler = scheduler
        
        # Default criterion is CrossEntropyLoss
        self.criterion = criterion or nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
        
        self.device = device
        self.model.to(self.device)
        
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.log_interval = log_interval
        
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
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()
            
            loss = self._compute_loss(batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
                self.training_history["learning_rate"].append(self.scheduler.get_last_lr()[0])
            
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (batch_idx + 1)
            
            # Update progress bar
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
            
            # Log at intervals
            if batch_idx % self.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Step {self.global_step} | Loss: {loss.item():.4f} | {elapsed:.2f}s elapsed")
            
            self.global_step += 1
        
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
                loss = self._compute_loss(batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(self.val_dataloader)
        self.training_history["val_loss"].append(avg_val_loss)
        
        return avg_val_loss
    
    def train(self, num_epochs: int, save_best_only: bool = True) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train for
            save_best_only: Whether to save only the best model
            
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
            
            # Validate
            if self.val_dataloader is not None:
                val_loss = self.validate()
                print(f"Validation Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    print(f"New best validation loss: {val_loss:.4f}")
                    self.save_checkpoint(is_best=True)
            else:
                # If no validation set, save based on training loss
                self.save_checkpoint(is_best=(epoch == num_epochs - 1))
            
            # Always save latest
            if not save_best_only:
                self.save_checkpoint(is_best=False)
        
        return self.training_history
    
    def save_checkpoint(self, is_best: bool = False) -> str:
        """
        Save a checkpoint of the model.
        
        Args:
            is_best: Whether this is the best model so far
            
        Returns:
            Path to the saved checkpoint
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save numbered checkpoint
        epoch_checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{self.current_epoch}.pt")
        torch.save(checkpoint, epoch_checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a checkpoint into the model and optimizer.
        
        Args:
            checkpoint_path: Path to the checkpoint to load
        """
        if not os.path.isfile(checkpoint_path):
            raise ValueError(f"No checkpoint found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        if "training_history" in checkpoint:
            self.training_history = checkpoint["training_history"]
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot the training history.
        
        Args:
            save_path: Path to save the plot to (optional)
        """
        plt.figure(figsize=(12, 4))
        
        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history["train_loss"], label="Training Loss")
        
        if self.val_dataloader is not None and self.training_history["val_loss"]:
            plt.plot(self.training_history["val_loss"], label="Validation Loss")
        
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        
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
        
        plt.show()
    
    def perplexity(self, loss: float) -> float:
        """
        Calculate perplexity from a loss value.
        
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