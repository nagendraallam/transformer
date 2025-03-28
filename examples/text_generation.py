"""
Example script that demonstrates how to generate text with a trained transformer model.
"""

import os
import sys
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import Transformer
from utils import get_tokenizer


def generate_text(
    checkpoint_path: str,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
):
    """
    Generate text using a trained transformer model.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        prompt: Input text to condition the generation
        max_length: Maximum length of the generated text
        temperature: Sampling temperature (higher = more random)
        top_k: Number of top tokens to consider for sampling
        
    Returns:
        Generated text
    """
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        # Use default config if not in checkpoint
        from configs.model_config import SMALL_CONFIG as config
    
    # Initialize tokenizer
    tokenizer = get_tokenizer(max_length=config.get("max_seq_length", 1024))
    
    # Initialize model
    model = Transformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=config.get("d_model", 512),
        num_heads=config.get("num_heads", 8),
        num_encoder_layers=config.get("num_encoder_layers", 6),
        num_decoder_layers=config.get("num_decoder_layers", 6),
        d_ff=config.get("d_ff", 2048),
        max_seq_length=config.get("max_seq_length", 1024),
        dropout=config.get("dropout", 0.1),
    )
    
    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Tokenize input text
    input_ids = tokenizer.encode(prompt, return_tensors="pt")["input_ids"].to(device)
    
    # Initialize output sequence with input
    output_sequence = input_ids
    
    print("Generating text...")
    
    # Generate text auto-regressively
    with torch.no_grad():
        for i in range(max_length):
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
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1} tokens...")
            
            # Break if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode the output sequence
    generated_text = tokenizer.decode(output_sequence[0])
    
    return generated_text


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate text with a trained transformer model")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="", help="Input text to condition the generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Number of top tokens to consider for sampling")
    
    args = parser.parse_args()
    
    # If no prompt is provided, ask for one
    if not args.prompt:
        args.prompt = input("Enter a prompt: ")
    
    # Generate text
    generated_text = generate_text(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    
    print("\nGenerated text:")
    print(generated_text) 