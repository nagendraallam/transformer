"""
GPT-2 model wrapper for pre-trained HuggingFace models.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
import math

class PreTrainedGPT2(nn.Module):
    """
    Wrapper for pre-trained GPT-2 model from HuggingFace.
    This maintains compatibility with the existing transformer structure
    while leveraging the pre-trained weights.
    """
    def __init__(
        self,
        pretrained_model_name="gpt2",
        vocab_size=50257,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,
        max_seq_length=1024,
        dropout=0.1,
        fine_tune=True,
    ):
        super(PreTrainedGPT2, self).__init__()
        
        # Load pre-trained model
        if pretrained_model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
            print(f"Loading pre-trained GPT-2 model: {pretrained_model_name}")
            self.model = GPT2LMHeadModel.from_pretrained(pretrained_model_name)
            
            # Resize token embeddings if vocab_size is different
            if vocab_size != self.model.config.vocab_size:
                print(f"Resizing token embeddings from {self.model.config.vocab_size} to {vocab_size}")
                self.model.resize_token_embeddings(vocab_size)
        else:
            # Initialize with custom config
            print(f"Initializing GPT-2 model with custom configuration")
            config = GPT2Config(
                vocab_size=vocab_size,
                n_embd=d_model,
                n_head=num_heads,
                n_layer=num_layers,
                n_positions=max_seq_length,
                n_ctx=max_seq_length,
                n_inner=d_ff,
                resid_pdrop=dropout,
                attn_pdrop=dropout,
                embd_pdrop=dropout,
            )
            self.model = GPT2LMHeadModel(config)
            
        # Whether to fine-tune or freeze the model
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
                
        # Store config for reference
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        
    def generate_square_subsequent_mask(self, size):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf')."""
        mask = (1 - torch.triu(torch.ones(size, size), diagonal=1)).bool()
        return mask
    
    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
        """
        Forward pass to maintain compatibility with existing code.
        Note: GPT-2 uses causal attention automatically, so tgt_mask is often not needed.
        
        Args:
            src: Source tensor of shape [batch_size, src_seq_len]
            tgt: Target tensor (optional, often same as src for language modeling)
            src_mask: Source mask (attention mask)
            tgt_mask: Target mask (not typically used for GPT-2)
            
        Returns:
            Output logits of shape [batch_size, seq_len, vocab_size]
        """
        # For language modeling, we typically use the same input for src and tgt
        input_ids = src
        attention_mask = src_mask if src_mask is not None else torch.ones_like(input_ids).bool()
        
        # Forward pass through the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Return logits
        return outputs.logits
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.model.gradient_checkpointing_enable()
        return self 