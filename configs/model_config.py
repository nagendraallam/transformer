"""
Configuration file for transformer model parameters.
"""

# Small model configuration (similar to GPT-2 small)
SMALL_CONFIG = {
    "d_model": 768,
    "num_heads": 8,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "d_ff": 2048,
    "max_seq_length": 2048,
    "dropout": 0.1,
    "vocab_size": 50257,  # GPT-2 tokenizer vocab size
    "activation": "gelu",
    "use_rotary_embeddings": True,
    "rope_theta": 10000,  # Base for rotary embeddings
}

# Tiny model configuration for testing
TINY_CONFIG = {
    "d_model": 128,
    "num_heads": 4,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "d_ff": 512,
    "max_seq_length": 128,
    "dropout": 0.1,
    "activation": "gelu",
    "use_rotary_embeddings": False,
}

# Medium model configuration
MEDIUM_CONFIG = {
    "d_model": 1024,
    "num_heads": 16,
    "num_encoder_layers": 12,
    "num_decoder_layers": 12,
    "d_ff": 4096,
    "max_seq_length": 2048,
    "dropout": 0.1,
    "vocab_size": 50257,
    "activation": "gelu",
    "use_rotary_embeddings": True,
    "rope_theta": 10000,
}

# Large model configuration
LARGE_CONFIG = {
    "d_model": 1280,
    "num_heads": 20,
    "num_encoder_layers": 24,
    "num_decoder_layers": 24,
    "d_ff": 5120,
    "max_seq_length": 2048,
    "dropout": 0.1,
    "vocab_size": 50257,
    "activation": "gelu",
    "use_rotary_embeddings": True,
    "rope_theta": 10000,
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 8,
    "learning_rate": 5e-5,
    "epochs": 10,
    "warmup_steps": 1000,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 1,
    "optimizer": "adamw",
    "scheduler": "linear_warmup",
}

# Configuration for tokenization
TOKENIZER_CONFIG = {
    "tokenizer_type": "bpe",  # byte-pair encoding
    "special_tokens": {
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "mask_token": "<mask>",
    }
}

# Inference configuration
INFERENCE_CONFIG = {
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "max_length": 100,
    "repetition_penalty": 1.0,
}

# Optimization configuration for lower memory usage
OPTIMIZATION_CONFIG = {
    "mixed_precision": False,
    "gradient_checkpointing": False,
    "bf16": False,
    "use_flash_attention": False,
}

# Default configuration to use
DEFAULT_CONFIG = SMALL_CONFIG 