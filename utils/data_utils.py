"""
Data utilities for loading and preprocessing text data.
"""

import os
import json
from typing import List, Dict, Tuple, Optional, Union, Iterator
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from .tokenizer import TransformerTokenizer


class TextDataset(Dataset):
    """Dataset for text sequences."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: TransformerTokenizer,
        max_length: int = 1024,
        return_tensors: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text sequences
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length
            return_tensors: Whether to return PyTorch tensors
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List[int]]]:
        text = self.texts[idx]
        
        # Encode the text
        encoding = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt" if self.return_tensors else None,
        )
        
        if self.return_tensors:
            # Remove batch dimension
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}
            
        return encoding


class SequencePairDataset(Dataset):
    """Dataset for sequence-to-sequence tasks."""
    
    def __init__(
        self,
        source_texts: List[str],
        target_texts: List[str],
        tokenizer: TransformerTokenizer,
        max_length: int = 1024,
        return_tensors: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            source_texts: List of source text sequences
            target_texts: List of target text sequences
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length
            return_tensors: Whether to return PyTorch tensors
        """
        assert len(source_texts) == len(target_texts), "Source and target texts must have the same length"
        
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors
    
    def __len__(self) -> int:
        return len(self.source_texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List[int]]]:
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]
        
        # Encode source text
        source_encoding = self.tokenizer.encode(
            source_text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt" if self.return_tensors else None,
        )
        
        # Encode target text
        target_encoding = self.tokenizer.encode(
            target_text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt" if self.return_tensors else None,
        )
        
        if self.return_tensors:
            # Remove batch dimension
            source_encoding = {k: v.squeeze(0) for k, v in source_encoding.items()}
            target_encoding = {k: v.squeeze(0) for k, v in target_encoding.items()}
        
        # Combine encodings
        encoding = {
            "input_ids": source_encoding["input_ids"],
            "attention_mask": source_encoding["attention_mask"],
            "decoder_input_ids": target_encoding["input_ids"],
            "decoder_attention_mask": target_encoding["attention_mask"],
            "labels": target_encoding["input_ids"],
        }
        
        return encoding


def load_json_dataset(
    file_path: str,
    tokenizer: TransformerTokenizer,
    max_length: int = 1024,
    return_tensors: bool = True,
    text_field: str = "text",
) -> TextDataset:
    """
    Load a dataset from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        tokenizer: Tokenizer for encoding texts
        max_length: Maximum sequence length
        return_tensors: Whether to return PyTorch tensors
        text_field: Name of the field containing the text in each JSON object
    
    Returns:
        TextDataset containing the texts
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        if isinstance(data[0], str):
            texts = data
        elif isinstance(data[0], dict) and text_field in data[0]:
            texts = [item[text_field] for item in data]
        else:
            raise ValueError(f"Could not extract texts from JSON data using field '{text_field}'")
    else:
        raise ValueError("JSON file should contain a list of strings or objects")
    
    return TextDataset(texts, tokenizer, max_length, return_tensors)


def load_text_file(
    file_path: str,
    tokenizer: TransformerTokenizer,
    max_length: int = 1024,
    return_tensors: bool = True,
) -> TextDataset:
    """
    Load a dataset from a text file, with one text per line.
    
    Args:
        file_path: Path to the text file
        tokenizer: Tokenizer for encoding texts
        max_length: Maximum sequence length
        return_tensors: Whether to return PyTorch tensors
    
    Returns:
        TextDataset containing the texts
    """
    with open(file_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    return TextDataset(texts, tokenizer, max_length, return_tensors)


def load_csv_dataset(
    file_path: str,
    tokenizer: TransformerTokenizer,
    max_length: int = 1024,
    return_tensors: bool = True,
    text_column: str = "text",
    label_column: Optional[str] = None,
) -> Union[TextDataset, Tuple[TextDataset, List]]:
    """
    Load a dataset from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        tokenizer: Tokenizer for encoding texts
        max_length: Maximum sequence length
        return_tensors: Whether to return PyTorch tensors
        text_column: Name of the column containing the text
        label_column: Name of the column containing the labels (optional)
    
    Returns:
        TextDataset containing the texts, optionally with labels
    """
    df = pd.read_csv(file_path)
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV file")
    
    texts = df[text_column].tolist()
    
    if label_column is not None:
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in CSV file")
        labels = df[label_column].tolist()
        return TextDataset(texts, tokenizer, max_length, return_tensors), labels
    
    return TextDataset(texts, tokenizer, max_length, return_tensors)


def load_sequence_pair_dataset(
    source_file: str,
    target_file: str,
    tokenizer: TransformerTokenizer,
    max_length: int = 1024,
    return_tensors: bool = True,
) -> SequencePairDataset:
    """
    Load a sequence-to-sequence dataset from two files.
    
    Args:
        source_file: Path to the file containing source sequences
        target_file: Path to the file containing target sequences
        tokenizer: Tokenizer for encoding texts
        max_length: Maximum sequence length
        return_tensors: Whether to return PyTorch tensors
    
    Returns:
        SequencePairDataset containing the source and target sequences
    """
    with open(source_file, "r", encoding="utf-8") as f:
        source_texts = [line.strip() for line in f if line.strip()]
    
    with open(target_file, "r", encoding="utf-8") as f:
        target_texts = [line.strip() for line in f if line.strip()]
    
    assert len(source_texts) == len(target_texts), "Source and target files must have the same number of lines"
    
    return SequencePairDataset(source_texts, target_texts, tokenizer, max_length, return_tensors)


def get_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        test_dataset: Test dataset (optional)
        batch_size: Batch size
        shuffle: Whether to shuffle the training data
        num_workers: Number of workers for data loading
    
    Returns:
        Dictionary containing the dataloaders
    """
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
    }
    
    if val_dataset is not None:
        dataloaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    
    if test_dataset is not None:
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    
    return dataloaders


def prepare_dummy_data(
    tokenizer: TransformerTokenizer,
    num_samples: int = 100,
    seq_length: int = 64,
    vocab_size: Optional[int] = None,
) -> TextDataset:
    """
    Create a dummy dataset for testing.
    
    Args:
        tokenizer: Tokenizer for encoding texts
        num_samples: Number of samples to generate
        seq_length: Length of each sequence
        vocab_size: Vocabulary size (defaults to tokenizer's vocab size)
    
    Returns:
        TextDataset containing the dummy data
    """
    if vocab_size is None:
        vocab_size = tokenizer.vocab_size
    
    # Generate random sequences of token IDs
    dummy_texts = []
    for _ in range(num_samples):
        token_ids = np.random.randint(1, vocab_size, size=seq_length).tolist()
        text = tokenizer.decode(token_ids)
        dummy_texts.append(text)
    
    return TextDataset(dummy_texts, tokenizer, seq_length) 