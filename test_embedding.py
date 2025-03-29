import torch
from model.transformer import Transformer
from transformers import GPT2Tokenizer

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f'Tokenizer vocab size: {tokenizer.vocab_size}')

# Load sample text
with open('data/sample.txt', 'r') as f:
    texts = [line.strip() for line in f if line.strip()]

# Tokenize with limited max length
encodings = tokenizer(texts, max_length=128, padding='max_length', truncation=True, return_tensors='pt')

# Initialize a small transformer model
model = Transformer(
    src_vocab_size=50257,  # Using exact tokenizer.vocab_size
    tgt_vocab_size=50257,  # Using exact tokenizer.vocab_size
    d_model=128,
    num_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_ff=512,
    max_seq_length=128,
    use_rotary_embeddings=False
)

# Test the encoder embedding specifically
try:
    print("Testing encoder embedding...")
    input_ids = encodings["input_ids"]
    embedded = model.encoder_embedding(input_ids)
    print(f"Encoder embedding output shape: {embedded.shape}")
    print("Encoder embedding works fine")
except Exception as e:
    print(f"Error with encoder embedding: {e}")

# Test the full model
try:
    print("\nTesting full model forward pass...")
    attention_mask = encodings["attention_mask"]
    outputs = model(src=input_ids, tgt=input_ids, src_mask=attention_mask)
    print(f"Model output shape: {outputs.shape}")
    print("Model forward pass works fine")
except Exception as e:
    print(f"Error with model forward pass: {e}") 