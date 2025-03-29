# Enhanced Transformer Implementation

This repository contains an enhanced implementation of the Transformer architecture, incorporating modern improvements and optimizations for better performance and efficiency.

## Key Features

### Architecture Improvements

- **RMSNorm**: Replaces traditional LayerNorm with Root Mean Square Layer Normalization for better training stability and efficiency
- **Rotary Position Embeddings (RoPE)**: Uses rotary embeddings as an alternative to traditional positional encodings for better handling of sequence information
- **Pre-LN Architecture**: Applies layer normalization before attention and feed-forward layers for improved training stability
- **GELU Activation**: Uses Gaussian Error Linear Units instead of ReLU for better performance
- **Efficient Attention**: Enhanced attention implementation with optimizations for better performance

### Training Enhancements

- **Mixed Precision Training**: Support for FP16/BF16 training to speed up computation and reduce memory usage
- **Gradient Checkpointing**: Option to trade computation for memory by recomputing intermediate activations during backpropagation
- **Flexible Checkpoint Saving**: Options for saving full checkpoints or weights-only models
- **Memory Efficiency**: Bias-free linear layers and other optimizations to reduce parameter count

### Inference Optimizations

- **Weight-Only Inference**: Ability to load just the model weights for efficient inference
- **Text Generation Controls**: Advanced sampling strategies including temperature, top-k, top-p, and repetition penalty
- **Multiple Precision Options**: Support for FP32, FP16, and BF16 inference

## Usage

### Training

```bash
python train.py --data data/sample.txt --model_size small --batch_size 4 --epochs 10 --device cuda --mixed_precision
```

Key training parameters:

- `--data`: Path to your training data
- `--model_size`: Size of model to use (small, medium, large)
- `--batch_size`: Batch size for training
- `--mixed_precision`: Enable mixed precision training (FP16)
- `--use_rotary_embeddings`: Use rotary embeddings instead of traditional positional encoding
- `--gradient_checkpointing`: Enable gradient checkpointing for memory efficiency
- `--gradient_accumulation_steps`: Accumulate gradients over multiple steps

### Inference

```bash
python inference.py --checkpoint checkpoints/best_model.pt --prompt "The transformer architecture is" --temperature 0.7 --max_length 100
```

Key inference parameters:

- `--checkpoint`: Path to the model checkpoint
- `--prompt`: Input text to start generation from
- `--temperature`: Controls randomness (lower = more deterministic)
- `--top_k`: Limits vocabulary to top k most likely tokens
- `--top_p`: Nucleus sampling threshold
- `--repetition_penalty`: Penalizes repeated tokens
- `--device`: Device to run on (cuda, cpu)

## Model Configurations

Three model sizes are available:

- **Small**: 125M parameters (similar to GPT-2 small)
  - Hidden size: 768
  - Attention heads: 12
  - Layers: 6+6 (encoder+decoder)
- **Medium**: 350M parameters
  - Hidden size: 1024
  - Attention heads: 16
  - Layers: 8+8 (encoder+decoder)
- **Large**: 750M parameters
  - Hidden size: 1280
  - Attention heads: 20
  - Layers: 12+12 (encoder+decoder)

## Implementation Details

The core architecture follows the original "Attention is All You Need" paper with modern enhancements:

1. **Tokenization**: Uses the Hugging Face Transformers library tokenizers
2. **Embedding**: Token embeddings + optional rotary position embeddings
3. **Encoder-Decoder**: Standard transformer with multi-head attention and feed-forward layers
4. **Normalization**: RMSNorm with Pre-LN ordering
5. **Inference**: Advanced sampling strategies for text generation

## Saving and Loading Models

For smaller file sizes and more efficient distribution, you can save just the model weights:

```python
# In your code
from utils.model_utils import save_model_weights_only

# Save in different formats
save_model_weights_only(model, "model_weights.pt")  # Regular FP32
save_model_weights_only(model, "model_weights_fp16.pt", half_precision=True)  # FP16
```

## Requirements

- PyTorch >= 1.10
- Transformers library
- NumPy
- tqdm

## Acknowledgments

This implementation was inspired by modern transformer architectures, particularly focusing on efficiency improvements from recent research.
