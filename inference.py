#!/usr/bin/env python
"""
Inference script for the transformer model.
"""

import os
import argparse
import torch
import torch.nn as nn
from typing import List
import numpy as np
import random

from model import Transformer
from utils import get_tokenizer
from configs.model_config import SMALL_CONFIG


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate(
    model: nn.Module,
    tokenizer,
    text: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    """
    Generate text using the transformer model.
    
    Args:
        model: Transformer model
        tokenizer: Tokenizer
        text: Input text (prompt)
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        device: Device to use
        
    Returns:
        Generated text
    """
    model.eval()
    
    # Tokenize input text
    input_ids = tokenizer.encode(text, return_tensors="pt")["input_ids"].to(device)
    
    # Initialize output sequence with input
    output_sequence = input_ids
    
    # Generate text auto-regressively
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs = model(
                src=output_sequence,
                tgt=output_sequence,
                src_mask=None,  # Will be created in the model
                tgt_mask=None,  # Will be created in the model
            )
            
            # Get the next token logits
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply top-k sampling
            if top_k > 0:
                # Get the top-k logits and their indices
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                
                # Create a mask for the top-k tokens
                mask = torch.zeros_like(next_token_logits).scatter_(1, top_k_indices, 1)
                next_token_logits = torch.where(mask == 1, next_token_logits, torch.tensor(-float("inf")).to(device))
            
            # Apply softmax to get probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the token to the output sequence
            output_sequence = torch.cat([output_sequence, next_token], dim=1)
            
            # Break if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode the output sequence
    generated_text = tokenizer.decode(output_sequence[0])
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate text with a transformer model")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_size", type=str, default="small", help="Model size: small, medium, large")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, default="", help="Prompt for generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu)")
    
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
        from configs.model_config import SMALL_CONFIG as model_config
    elif args.model_size.lower() == "medium":
        from configs.model_config import MEDIUM_CONFIG as model_config
    elif args.model_size.lower() == "large":
        from configs.model_config import LARGE_CONFIG as model_config
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
    
    # Initialize tokenizer
    tokenizer = get_tokenizer(max_length=model_config["max_seq_length"])
    
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
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    print(f"Loaded model from {args.checkpoint}")
    
    # Generate text
    if not args.prompt:
        args.prompt = input("Enter a prompt: ")
    
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        text=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    
    print("\nGenerated text:")
    print(generated_text)


if __name__ == "__main__":
    main() 