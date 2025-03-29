import torch
from transformers import GPT2Tokenizer

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load sample text
with open('data/sample.txt', 'r') as f:
    texts = [line.strip() for line in f if line.strip()]

# Tokenize with limited max length
encodings = tokenizer(texts, max_length=128, padding='max_length', truncation=True, return_tensors='pt')

# Print info about the encodings
print(f'Input IDs shape: {encodings["input_ids"].shape}')
print(f'Min index: {encodings["input_ids"].min()}')
print(f'Max index: {encodings["input_ids"].max()}')
print(f'Vocab size: {tokenizer.vocab_size}')

# Print the first few token IDs from the first sentence
first_sentence_ids = encodings["input_ids"][0][:20].tolist()
print(f'First 20 token IDs from first sentence: {first_sentence_ids}')

# Test embedding layer
embed = torch.nn.Embedding(tokenizer.vocab_size, 128)
try:
    embed_output = embed(encodings["input_ids"])
    print(f'Embedding output shape: {embed_output.shape}')
    print("Embedding layer works fine")
except Exception as e:
    print(f'Error with embedding layer: {e}') 