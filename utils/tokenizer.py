"""
Tokenizer utilities for text preprocessing.
"""

from typing import List, Dict, Union, Optional
from transformers import AutoTokenizer
import os
import torch
import json


class TransformerTokenizer:
    """Wrapper around HuggingFace tokenizers for our transformer model."""
    
    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        max_length: int = 1024,
        use_pretrained: bool = True,
        special_tokens: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the tokenizer.
        
        Args:
            tokenizer_name: Name of the pretrained tokenizer or path to a saved tokenizer
            max_length: Maximum sequence length
            use_pretrained: Whether to use a pretrained tokenizer
            special_tokens: Dictionary of special tokens to add to the tokenizer
        """
        self.max_length = max_length
        
        if use_pretrained:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            if os.path.exists(tokenizer_name):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
            else:
                raise ValueError(f"Tokenizer {tokenizer_name} not found")
        
        # Set pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            # For models like GPT2 that don't have a pad token by default
            # Set pad_token to eos_token for convenience
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Add special tokens if provided
        if special_tokens:
            self.tokenizer.add_special_tokens(special_tokens)
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text using the tokenizer.
        
        Args:
            text: Input text or list of texts
            add_special_tokens: Whether to add special tokens
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length (overrides the default)
            return_tensors: Format of the returned tensors ("pt" for PyTorch)
            
        Returns:
            Dictionary of encoded inputs
        """
        # Use the provided max_length if given, otherwise use the default
        max_len = max_length if max_length is not None else self.max_length
        
        return self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_len,
            return_tensors=return_tensors,
        )
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List or tensor of token IDs
            skip_special_tokens: Whether to skip special tokens in the decoded output
            
        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_decode(
        self,
        sequences: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode a batch of token IDs back to texts.
        
        Args:
            sequences: Batch of token ID sequences
            skip_special_tokens: Whether to skip special tokens in the decoded output
            
        Returns:
            List of decoded texts
        """
        return self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)
    
    def save_pretrained(self, save_directory: str):
        """
        Save the tokenizer to disk.
        
        Args:
            save_directory: Directory to save the tokenizer to
        """
        self.tokenizer.save_pretrained(save_directory)
        
        # Save additional config
        with open(os.path.join(save_directory, "tokenizer_config.json"), "w") as f:
            json.dump({"max_length": self.max_length}, f)
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size of the tokenizer."""
        return len(self.tokenizer)
    
    @property
    def pad_token_id(self) -> int:
        """Get the ID of the padding token."""
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        """Get the ID of the end-of-sequence token."""
        return self.tokenizer.eos_token_id
    
    @property
    def bos_token_id(self) -> int:
        """Get the ID of the beginning-of-sequence token."""
        return self.tokenizer.bos_token_id


def get_tokenizer(
    tokenizer_name: str = "gpt2",
    max_length: int = 1024,
    use_pretrained: bool = True,
    special_tokens: Optional[Dict[str, str]] = None,
) -> TransformerTokenizer:
    """
    Helper function to get a preconfigured tokenizer.
    
    Args:
        tokenizer_name: Name of the pretrained tokenizer or path to a saved tokenizer
        max_length: Maximum sequence length
        use_pretrained: Whether to use a pretrained tokenizer
        special_tokens: Dictionary of special tokens to add to the tokenizer
        
    Returns:
        Initialized tokenizer
    """
    return TransformerTokenizer(
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        use_pretrained=use_pretrained,
        special_tokens=special_tokens,
    ) 