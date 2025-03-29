# Enhanced Transformer Commands Reference

This document provides a comprehensive reference for all command-line interfaces in the Enhanced Transformer project. Each command's parameters are explained in detail with their purpose and usage.

## Table of Contents

1. [Model Training Commands](#model-training-commands)
   - [`train.py` - Train a Transformer Model](#trainpy---train-a-transformer-model)
   - [`fine_tune_gpt2.py` - Fine-tune a Pre-trained GPT-2 Model](#fine_tune_gpt2py---fine-tune-a-pre-trained-gpt-2-model)

2. [Inference Commands](#inference-commands)
   - [`inference.py` - Generate Text with a Transformer Model](#inferencepy---generate-text-with-a-transformer-model)
   - [`gpt2_inference.py` - Generate Text with a Fine-tuned GPT-2 Model](#gpt2_inferencepy---generate-text-with-a-fine-tuned-gpt-2-model)

## Model Training Commands

### `train.py` - Train a Transformer Model

Trains a transformer model from scratch on custom text data.

#### Usage

```bash
python train.py --data data/sample.txt --model_size small --batch_size 8 --epochs 10
```

#### Parameters

##### Data Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data` | string | None | Path to data file (text, JSON, or CSV) |
| `--val_split` | float | 0.1 | Validation split ratio |
| `--dummy_data` | flag | False | Use dummy data for testing |
| `--dummy_samples` | int | 1000 | Number of dummy samples when using dummy data |
| `--dummy_seq_length` | int | 64 | Sequence length for dummy data |

##### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_size` | string | "small" | Model size: "small", "medium", or "large" |
| `--max_seq_length` | int | 2048 | Maximum sequence length |
| `--use_rotary_embeddings` | flag | False | Use rotary embeddings |
| `--activation` | string | "gelu" | Activation function ("gelu", "relu") |

##### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--batch_size` | int | 8 | Batch size for training |
| `--epochs` | int | 10 | Number of training epochs |
| `--learning_rate` | float | 5e-5 | Learning rate |
| `--warmup_steps` | int | 1000 | Warmup steps for scheduler |
| `--weight_decay` | float | 0.01 | Weight decay for optimizer |
| `--gradient_accumulation_steps` | int | 1 | Gradient accumulation steps |

##### Optimization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--mixed_precision` | flag | False | Use mixed precision training |
| `--gradient_checkpointing` | flag | False | Use gradient checkpointing |
| `--bf16` | flag | False | Use bfloat16 precision instead of float16 |

##### Other Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--seed` | int | 42 | Random seed for reproducibility |
| `--device` | string | None | Device to use ("cuda", "cpu") |
| `--save_dir` | string | "checkpoints" | Directory to save checkpoints |
| `--save_best_only` | flag | False | Only save the best model based on validation loss |
| `--log_interval` | int | 100 | Logging interval in training steps |

### `fine_tune_gpt2.py` - Fine-tune a Pre-trained GPT-2 Model

Fine-tunes a pre-trained GPT-2 model on custom text data.

#### Usage

```bash
python fine_tune_gpt2.py --data data/sample.txt --model_size small --batch_size 4 --epochs 3
```

#### Parameters

##### Data Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data` | string | None | Path to training data file |
| `--dummy_data` | flag | False | Use dummy data for testing |
| `--dummy_samples` | int | 100 | Number of dummy samples when using dummy data |
| `--dummy_seq_length` | int | 64 | Length of dummy sequences |
| `--val_split` | float | 0.1 | Validation split ratio |
| `--max_seq_length` | int | 1024 | Maximum sequence length |

##### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_size` | string | "small" | GPT-2 model size: "small" (base gpt2), "medium", "large", or "xl" |
| `--fine_tune_all` | flag | False | Fine-tune all parameters (otherwise only fine-tunes top layers) |

##### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--batch_size` | int | 4 | Batch size for training |
| `--epochs` | int | 3 | Number of training epochs |
| `--learning_rate` | float | 5e-5 | Learning rate |
| `--weight_decay` | float | 0.01 | Weight decay for optimizer |
| `--warmup_steps` | int | 200 | Number of warmup steps |
| `--gradient_accumulation_steps` | int | 1 | Gradient accumulation steps |
| `--log_interval` | int | 10 | Logging interval in training steps |
| `--save_dir` | string | "checkpoints" | Directory to save model checkpoints |
| `--save_best_only` | flag | False | Save only the best model based on validation loss |

##### Mixed Precision Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--mixed_precision` | flag | False | Use mixed precision training |
| `--bf16` | flag | False | Use bfloat16 precision |
| `--gradient_checkpointing` | flag | False | Use gradient checkpointing |

##### Other Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--seed` | int | 42 | Random seed for reproducibility |
| `--device` | string | None | Device to use ("cuda" or "cpu") |

## Inference Commands

### `inference.py` - Generate Text with a Transformer Model

Generates text using a trained transformer model.

#### Usage

```bash
python inference.py --checkpoint checkpoints/best_model.pt --prompt "The transformer model"
```

#### Parameters

##### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--checkpoint` | string | Required | Path to model checkpoint |
| `--prompt` | string | Required | Prompt for text generation |

##### Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--max_length` | int | 100 | Maximum number of tokens to generate |
| `--temperature` | float | 0.7 | Temperature for sampling (higher = more random) |
| `--top_k` | int | 50 | Top-k sampling parameter (limits to top k tokens) |
| `--top_p` | float | 0.9 | Top-p (nucleus) sampling parameter (limits to top p probability mass) |
| `--repetition_penalty` | float | 1.0 | Penalty for repeating tokens (> 1.0 reduces repetition) |

##### Other Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--seed` | int | 42 | Random seed for reproducibility |
| `--device` | string | None | Device to use ("cuda", "cpu") |

### `gpt2_inference.py` - Generate Text with a Fine-tuned GPT-2 Model

Generates text using a fine-tuned GPT-2 model.

#### Usage

```bash
python gpt2_inference.py --checkpoint checkpoints/gpt2_model.pt --prompt "The language model"
```

#### Parameters

##### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--checkpoint` | string | None | Path to model checkpoint (optional) |
| `--model_size` | string | "small" | GPT-2 model size if no checkpoint provided: "small", "medium", "large", "xl" |
| `--prompt` | string | Required | Prompt for text generation |

##### Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--max_length` | int | 100 | Maximum number of tokens to generate |
| `--temperature` | float | 0.7 | Temperature for sampling (higher = more random) |
| `--top_k` | int | 50 | Top-k sampling parameter (limits to top k tokens) |
| `--top_p` | float | 0.9 | Top-p (nucleus) sampling parameter (limits to top p probability mass) |
| `--repetition_penalty` | float | 1.0 | Penalty for repeating tokens (> 1.0 reduces repetition) |

##### Other Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--seed` | int | 42 | Random seed for reproducibility |
| `--device` | string | None | Device to use ("cuda", "cpu") | 