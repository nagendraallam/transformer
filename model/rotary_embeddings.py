"""
Rotary Position Embeddings implementation.
Based on RoFormer paper: https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RotaryEmbedding(nn.Module):
    """
    Rotary positional embeddings (RoPE) for transformers.
    This implements the rotation in the complex plane as described in the 
    RoFormer paper.
    """
    
    def __init__(
        self, 
        dim: int, 
        max_position_embeddings: int = 2048, 
        base: int = 10000,
        device=None
    ):
        """
        Initialize the rotary embeddings.
        
        Args:
            dim: Dimension of each head
            max_position_embeddings: Maximum sequence length
            base: Base value for frequencies
            device: Device to place the embeddings on
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Create buffer for cos and sin values
        self.inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", self.inv_freq)
        
        # Cache for cos and sin values
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        
    def _update_cos_sin_cache(self, seq_len: int, device=None):
        """
        Update the cached cos and sin values.
        
        Args:
            seq_len: Sequence length to generate values for
            device: Device to place the values on
        """
        # Check if we need to update the cache
        if (
            self._cos_cached is None or 
            self._sin_cached is None or 
            self._cos_cached.device != device or 
            self._cos_cached.shape[0] < seq_len
        ):
            # Get correct device
            device = self.inv_freq.device if device is None else device
            
            # Generate position indices
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            
            # Compute position frequencies
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            
            # Compute cos and sin
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(device)
            self._sin_cached = emb.sin().to(device)
            
    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor = None):
        """
        Apply rotary embeddings to queries and keys.
        
        Args:
            q: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
            positions: Position ids (optional, defaults to 0..seq_len-1)
            
        Returns:
            Tuple of (q_rotated, k_rotated)
        """
        # Get sequence length and device
        device = q.device
        seq_len = max(q.shape[-2], k.shape[-2])
        
        # Update the cache
        self._update_cos_sin_cache(seq_len, device)
        
        # Get cos and sin values for the sequence
        cos = self._cos_cached[:seq_len]
        sin = self._sin_cached[:seq_len]
        
        # Apply position offsets if provided
        if positions is not None:
            cos = cos.index_select(0, positions.reshape(-1)).reshape(*positions.shape, -1)
            sin = sin.index_select(0, positions.reshape(-1)).reshape(*positions.shape, -1)
            
            # Add dimensions for heads if needed
            # cos, sin: [batch_size, seq_len, dim]
            # q, k: [batch_size, n_heads, seq_len, dim]
            if q.ndim == 4 and cos.ndim == 3:
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
        
        # Apply rotary embeddings
        return self.apply_rotary_emb(q, k, cos, sin)
    
    def apply_rotary_emb(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """
        Apply rotary embeddings to query and key tensors.
        
        Args:
            q: Query tensor
            k: Key tensor
            cos: Cosine values for rotation
            sin: Sine values for rotation
            
        Returns:
            Tuple of (q_rotated, k_rotated)
        """
        # Reshape for easier handling
        q_dim = q.shape[-1]
        k_dim = k.shape[-1]
        
        # Split each head dim into half for complex number representation
        q_real, q_imag = q[..., :q_dim//2], q[..., q_dim//2:]
        k_real, k_imag = k[..., :k_dim//2], k[..., k_dim//2:]
        
        # Apply complex multiplication:
        # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        # where the rotation is (cos + sin*i)
        
        # Truncate cos and sin to match q and k dimensions
        cos_q = cos[..., :q_dim//2]
        sin_q = sin[..., :q_dim//2]
        cos_k = cos[..., :k_dim//2]
        sin_k = sin[..., :k_dim//2]
        
        # Apply rotation to q and k
        q_out_real = q_real * cos_q - q_imag * sin_q
        q_out_imag = q_real * sin_q + q_imag * cos_q
        k_out_real = k_real * cos_k - k_imag * sin_k
        k_out_imag = k_real * sin_k + k_imag * cos_k
        
        # Concatenate real and imaginary parts
        q_out = torch.cat([q_out_real, q_out_imag], dim=-1)
        k_out = torch.cat([k_out_real, k_out_imag], dim=-1)
        
        return q_out, k_out 