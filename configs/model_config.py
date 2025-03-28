"""
Configuration file for transformer model parameters.
"""

# Small model configuration (similar to GPT-2 small)
SMALL_CONFIG = {
    "d_model": 768,
    "num_heads": 12,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "d_ff": 3072,
    "max_seq_length": 1024,
    "dropout": 0.1,
    "vocab_size": 50257,  # GPT-2 tokenizer vocab size
}

# Medium model configuration
MEDIUM_CONFIG = {
    "d_model": 1024,
    "num_heads": 16,
    "num_encoder_layers": 8,
    "num_decoder_layers": 8,
    "d_ff": 4096,
    "max_seq_length": 1024,
    "dropout": 0.1,
    "vocab_size": 50257,
}

# Large model configuration
LARGE_CONFIG = {
    "d_model": 1280,
    "num_heads": 20,
    "num_encoder_layers": 12,
    "num_decoder_layers": 12,
    "d_ff": 5120,
    "max_seq_length": 1024,
    "dropout": 0.1,
    "vocab_size": 50257,
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 8,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_steps": 10000,
    "max_steps": 100000,
    "save_steps": 10000,
    "eval_steps": 1000,
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

# Default configuration to use
DEFAULT_CONFIG = SMALL_CONFIG 