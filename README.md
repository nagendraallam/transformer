# Enhanced Transformer

This project implements an enhanced transformer model with optimizations for training efficiency. It includes several optimization techniques like gradient checkpointing, mixed precision training, flash attention, and more.

## Features

- Transformer encoder-decoder architecture
- Efficient attention implementations including Flash Attention (when available)
- Gradient checkpointing for memory efficiency
- Mixed precision training (fp16/bf16)
- RMSNorm layer normalization
- Rotary positional embeddings
- Lion optimizer support
- Bucketing for efficient sequence packing
- Model compilation with torch.compile (experimental)

## Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # For bash/zsh
# For fish shell:
source venv/bin/activate.fish

# Install dependencies
pip install -r requirements.txt

# Install optional optimizers
pip install lion-pytorch
```

## Usage

### Training

```bash
# Basic training
python train.py --data data/sample.txt --model_size tiny --batch_size 4 --epochs 3

# Train with optimizations
python train.py --data data/sample.txt --model_size small --batch_size 4 --epochs 3 \
    --device cuda --optimizer lion --mixed_precision --gradient_checkpointing \
    --save_dir checkpoints_optimized

# Train with all optimizations (requires more GPU memory)
python train.py --data data/sample.txt --model_size small --batch_size 4 --epochs 3 \
    --device cuda --optimizer lion --mixed_precision --flash_attention \
    --gradient_checkpointing --compile --save_dir checkpoints_optimized
```

### Available Arguments

- `--data`: Path to the training data file
- `--model_size`: Model size (tiny, small, medium, large)
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--device`: Device to train on (cpu or cuda)
- `--optimizer`: Optimizer to use (adam, adamw, lion)
- `--learning_rate`: Learning rate
- `--mixed_precision`: Enable mixed precision training
- `--flash_attention`: Use flash attention when available
- `--gradient_checkpointing`: Enable gradient checkpointing
- `--compile`: Use torch.compile for faster training
- `--bucketing`: Use bucketing for efficient sequence packing
- `--num_workers`: Number of workers for data loading
- `--save_dir`: Directory to save model checkpoints

## Model Configurations

- **Tiny**: 128 dimensions, 4 heads, 2 layers (for testing)
- **Small**: 768 dimensions, 8 heads, 12 layers
- **Medium**: 1024 dimensions, 16 heads, 12 layers
- **Large**: 1280 dimensions, 20 heads, 24 layers

## Known Issues and Workarounds

- When using mixed precision training, you may see warnings about "No inf checks were recorded for this optimizer". This is expected and automatically handled by falling back to regular optimizer steps.
- Torch compile may not work properly with gradient checkpointing - consider using them separately.
- For large models, reduce batch size and use gradient checkpointing to avoid out-of-memory errors.
- Flash attention may not be available on all hardware - the model will automatically fall back to standard attention when not available.

## License

MIT
