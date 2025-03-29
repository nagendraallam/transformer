#!/usr/bin/env python
"""
Inference script for the transformer model.
"""

import os
import argparse
import torch
import numpy as np
import random
from typing import List, Optional, Union, Dict, Any

from model import Transformer
from utils import get_tokenizer
from utils.model_utils import load_model_weights, print_model_size
from configs.model_config import SMALL_CONFIG, MEDIUM_CONFIG, LARGE_CONFIG, INFERENCE_CONFIG, TINY_CONFIG


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_checkpoint(checkpoint_path: str, device: str):
    """
    Load a model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to
        
    Returns:
        Loaded model and model configuration
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Try to load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine if this is a full checkpoint or just weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full checkpoint with model state, optimizer state, etc.
        model_state = checkpoint['model_state_dict']
        
        # Check if config is stored in the checkpoint
        if 'config' in checkpoint:
            model_config = checkpoint['config']
        else:
            # Assume small config if not specified
            print("No model configuration found in checkpoint. Using default small config.")
            model_config = SMALL_CONFIG
    else:
        # Just the model weights
        model_state = checkpoint
        # Use TINY_CONFIG for our optimized model
        print("Loading weights-only checkpoint. Using TINY_CONFIG.")
        model_config = TINY_CONFIG.copy()
        # The checkpoint was trained with max_seq_length 2048, so we need to keep it
        model_config['max_seq_length'] = 2048
    
    # Create model with the appropriate configuration
    model = Transformer(
        src_vocab_size=model_config.get('vocab_size', 50257),
        tgt_vocab_size=model_config.get('vocab_size', 50257),
        d_model=model_config.get('d_model', 768),
        num_heads=model_config.get('num_heads', 12),
        num_encoder_layers=model_config.get('num_encoder_layers', 6),
        num_decoder_layers=model_config.get('num_decoder_layers', 6),
        d_ff=model_config.get('d_ff', 3072),
        max_seq_length=model_config.get('max_seq_length', 2048),
        dropout=0.0,  # No dropout during inference
        activation=model_config.get('activation', 'gelu'),
        use_rotary_embeddings=model_config.get('use_rotary_embeddings', True),
    )
    
    # Load the model weights
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"Successfully loaded model")
    print_model_size(model)
    
    return model, model_config


def generate_text(
    model: Transformer,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    device: str = 'cpu',
) -> str:
    """
    Generate text using the transformer model.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer to use
        prompt: The prompt to start generation from
        max_length: Maximum number of tokens to generate
        temperature: Temperature for sampling
        top_k: Number of highest probability tokens to consider for each step
        top_p: Probability threshold for nucleus sampling
        repetition_penalty: Penalty for repeating tokens
        device: Device to run inference on
        
    Returns:
        Generated text
    """
    model.eval()  # Ensure model is in evaluation mode
    
    # Encode the prompt
    prompt_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Create attention mask for the prompt tokens
    prompt_mask = torch.ones_like(prompt_tokens, dtype=torch.bool)
    
    # Set up for generation
    generated_tokens = prompt_tokens.clone()
    generated_mask = prompt_mask.clone()
    
    # Create a causal mask for the entire sequence
    causal_mask = model.generate_square_subsequent_mask(max_length).to(device)
    
    with torch.no_grad():
        # Generate tokens one by one
        for _ in range(max_length):
            # Get current sequence length
            seq_len = generated_tokens.shape[1]
            
            # Forward pass through encoder and decoder
            # Use the prompt as both source and target for a language model
            logits = model(
                generated_tokens,
                generated_tokens,
                src_mask=None,
                tgt_mask=None,
            )
            
            # Get the next token logits (last position in sequence)
            next_token_logits = logits[:, -1, :].squeeze(0)
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_tokens[0].tolist()):
                    if next_token_logits[token_id] < 0:
                        next_token_logits[token_id] *= repetition_penalty
                    else:
                        next_token_logits[token_id] /= repetition_penalty
            
            # Filter with top-k
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Filter with top-p (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).unsqueeze(0)
            
            # Check if we've hit the end of text token
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Append to generated tokens
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            # Append to mask
            next_token_mask = torch.ones_like(next_token, dtype=torch.bool)
            generated_mask = torch.cat([generated_mask, next_token_mask], dim=1)
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Run inference with a transformer model")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for text generation")
    
    # Generation parameters
    parser.add_argument("--max_length", type=int, default=INFERENCE_CONFIG["max_length"], help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=INFERENCE_CONFIG["temperature"], help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=INFERENCE_CONFIG["top_k"], help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=INFERENCE_CONFIG["top_p"], help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=INFERENCE_CONFIG["repetition_penalty"], help="Penalty for repeating tokens")
    
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
    
    # Load model from checkpoint
    model, model_config = load_model_checkpoint(args.checkpoint, device)
    
    # Initialize tokenizer
    tokenizer = get_tokenizer(max_length=model_config.get('max_seq_length', 2048))
    
    # Print generation parameters
    print("\nGeneration Parameters:")
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Max Length: {args.max_length}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Repetition Penalty: {args.repetition_penalty}")
    
    # Generate text
    print("\nGenerating text...")
    
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=device,
    )
    
    print("\nGenerated Text:")
    print("-" * 80)
    print(generated_text)
    print("-" * 80)


if __name__ == "__main__":
    main() 