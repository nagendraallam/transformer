# Enhanced Transformer Usage Guide

This document provides instructions on how to use the Enhanced Transformer implementation for training and inference.

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training a Model

### Using the Training Script

The simplest way to train a model is to use the `train.py` script:

```bash
python train.py --data data/sample.txt --model_size small --batch_size 4 --epochs 10
```

Key parameters:

- `--data`: Path to your training data file (text, JSON, or CSV)
- `--model_size`: Size of the model (`small`, `medium`, or `large`)
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--learning_rate`: Learning rate (default: 5e-5)
- `--max_seq_length`: Maximum sequence length (default: 1024)
- `--dummy_data`: Use dummy data for testing (no real training)

For more options, run `python train.py --help`.

### Using Custom Data

Your data file can be in the following formats:

- **Text file**: One text example per line
- **JSON file**: A list of strings or objects with a "text" field
- **CSV file**: A file with a "text" column

### Using the Python API

You can also train a model programmatically:

```python
from model import Transformer
from utils import get_tokenizer, load_text_file, get_dataloaders, TransformerTrainer

# Initialize tokenizer
tokenizer = get_tokenizer()

# Load data
dataset = load_text_file("data/sample.txt", tokenizer)

# Create dataloaders
dataloaders = get_dataloaders(train_dataset=dataset, val_split=0.1, batch_size=4)

# Initialize model
model = Transformer(
    src_vocab_size=tokenizer.vocab_size,
    tgt_vocab_size=tokenizer.vocab_size,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    max_seq_length=1024,
    dropout=0.1,
)

# Initialize trainer and train
trainer = TransformerTrainer(
    model=model,
    train_dataloader=dataloaders["train"],
    val_dataloader=dataloaders["val"],
)
trainer.train(num_epochs=10)
```

## Inference (Text Generation)

### Using the Inference Script

After training a model, you can generate text using the `inference.py` script:

```bash
python inference.py --checkpoint checkpoints/best_model.pt --prompt "The transformer model"
```

Key parameters:

- `--checkpoint`: Path to the trained model checkpoint
- `--prompt`: Text prompt to start generation
- `--max_length`: Maximum length of generated text (default: 100)
- `--temperature`: Sampling temperature (default: 1.0) - higher means more random
- `--top_k`: Top-k sampling parameter (default: 50)

### Using the Python API

You can also generate text programmatically:

```python
from inference import generate
from model import Transformer
from utils import get_tokenizer

# Load model and generate text
generated_text = generate(
    checkpoint_path="checkpoints/best_model.pt",
    prompt="The transformer model",
    max_length=100,
    temperature=0.8,
    top_k=50,
)

print(generated_text)
```

## Example Use Cases

### Training a Small Model

```bash
python train.py --data data/sample.txt --model_size small --batch_size 4 --epochs 5
```

### Training with Custom Parameters

```bash
python train.py --data data/sample.txt --model_size medium --batch_size 2 --epochs 10 --learning_rate 1e-5 --max_seq_length 512
```

### Text Generation with Lower Temperature

```bash
python inference.py --checkpoint checkpoints/best_model.pt --prompt "Transformers are" --temperature 0.7 --max_length 200
```

## Model Configurations

The project comes with three pre-defined model configurations:

1. **Small** (default): ~125M parameters

   - 768 hidden size, 12 attention heads, 6 layers

2. **Medium**: ~350M parameters

   - 1024 hidden size, 16 attention heads, 8 layers

3. **Large**: ~750M parameters
   - 1280 hidden size, 20 attention heads, 12 layers

You can modify these configurations in `configs/model_config.py`.

## Saving and Loading Models

Models are automatically saved during training to the `checkpoints` directory:

- `latest_checkpoint.pt`: The most recent model state
- `best_model.pt`: The model with the lowest validation loss
- `checkpoint_epoch_{N}.pt`: Model after epoch N

To load a model for further training:

```python
from utils import TransformerTrainer

trainer = TransformerTrainer(model, train_dataloader)
trainer.load_checkpoint("checkpoints/best_model.pt")
trainer.train(num_epochs=5)  # Continue training
```

## Monitoring Training

During training, loss values are logged to the console. After training completes, a training history plot is saved to `checkpoints/training_history.png`.

## Additional Examples

See the `examples` directory for additional example scripts:

- `examples/custom_training.py`: Example of custom training
- `examples/text_generation.py`: Example of text generation
