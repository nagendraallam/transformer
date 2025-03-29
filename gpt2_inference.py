#!/usr/bin/env python
"""
Inference script for fine-tuned GPT-2 models.
"""

import os
import argparse
import torch
import numpy as np
import random
from typing import List, Optional

from model import PreTrainedGPT2
from utils.tokenizer import get_tokenizer


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_checkpoint(checkpoint_path: str, device: str, model_size: str = "small"):
    """
    Load a GPT-2 model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to
        model_size: Size of the GPT-2 model
        
    Returns:
        Loaded model
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Determine model name based on size
    model_name_mapping = {
        "small": "gpt2",
        "medium": "gpt2-medium",
        "large": "gpt2-large",
        "xl": "gpt2-xl"
    }
    pretrained_model_name = model_name_mapping[model_size]
    
    # Initialize the model architecture
    model = PreTrainedGPT2(pretrained_model_name=pretrained_model_name)
    
    # Try to load the checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Determine if this is a full checkpoint or just weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Full checkpoint with model state, optimizer state, etc.
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Just the model weights
            model.load_state_dict(checkpoint)
            
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        print(f"Successfully loaded model from checkpoint")
        
        # Get parameter count
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model has {param_count:,} parameters")
        
        return model
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Falling back to pre-trained model without fine-tuning")
        model = model.to(device)
        model.eval()
        return model


def generate_text(
    model,
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
    Generate text using a GPT-2 model.
    
    Args:
        model: The GPT-2 model
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
    inputs = tokenizer.encode(
        prompt, 
        return_tensors='pt',
        add_special_tokens=True
    ).to(device)
    
    # Set attention mask for the prompt tokens
    attention_mask = torch.ones_like(inputs)
    
    # Generate output with huggingface generation method
    output = model.model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=len(inputs[0]) + max_length,  # Don't repeat the prompt in max_length
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate text with a fine-tuned GPT-2 model")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "medium", "large", "xl"],
                       help="GPT-2 model size if no checkpoint provided")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for text generation")
    
    # Generation parameters
    parser.add_argument("--max_length", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Penalty for repeating tokens")
    
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
    
    # Load model
    if args.checkpoint:
        model = load_model_checkpoint(args.checkpoint, device, args.model_size)
    else:
        print(f"No checkpoint provided. Using pre-trained GPT-2 {args.model_size} model")
        model_name_mapping = {
            "small": "gpt2",
            "medium": "gpt2-medium", 
            "large": "gpt2-large",
            "xl": "gpt2-xl"
        }
        pretrained_model_name = model_name_mapping[args.model_size]
        model = PreTrainedGPT2(pretrained_model_name=pretrained_model_name)
        model = model.to(device)
        model.eval()
    
    # Initialize tokenizer
    tokenizer = get_tokenizer(max_length=2048)
    
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