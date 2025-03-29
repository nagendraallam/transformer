"""
Flash Attention implementation for improved performance.
Based on FlashAttention paper: https://arxiv.org/abs/2205.14135
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Check if FlashAttention is available
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("FlashAttention not available. Using standard attention instead.")


class FlashAttention(nn.Module):
    """
    Flash Attention implementation for efficient attention computation.
    This optimizes both memory usage and compute speed.
    """
    
    def __init__(self, dropout_p=0.0):
        """
        Initialize the Flash Attention module.
        
        Args:
            dropout_p: Dropout probability
        """
        super().__init__()
        self.dropout_p = dropout_p
        
        # Check if we can use torch's built-in implementation
        self.use_builtin_flash_attn = hasattr(F, 'scaled_dot_product_attention')
    
    def _chunk_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: int = 1024
    ) -> torch.Tensor:
        """
        Compute attention in chunks to reduce memory usage.
        
        Args:
            q: Query tensor of shape [batch_size, num_heads, seq_len_q, head_dim]
            k: Key tensor of shape [batch_size, num_heads, seq_len_k, head_dim]
            v: Value tensor of shape [batch_size, num_heads, seq_len_k, head_dim]
            mask: Attention mask of shape [batch_size, 1, seq_len_q, seq_len_k]
            chunk_size: Size of chunks for computation
            
        Returns:
            Output tensor of shape [batch_size, num_heads, seq_len_q, head_dim]
        """
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        seq_len_k = k.shape[2]
        
        # Initialize output
        output = torch.zeros_like(q)
        
        # Process query sequence in chunks
        for i in range(0, seq_len_q, chunk_size):
            q_chunk = q[:, :, i:i+chunk_size]
            
            # Process key/value sequence in chunks
            chunk_attn_weights = []
            chunk_context = []
            
            for j in range(0, seq_len_k, chunk_size):
                k_chunk = k[:, :, j:j+chunk_size]
                v_chunk = v[:, :, j:j+chunk_size]
                
                # Compute attention scores for this chunk pair
                # [batch_size, num_heads, chunk_q, chunk_k]
                attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / math.sqrt(head_dim)
                
                # Apply mask if provided
                if mask is not None:
                    chunk_mask = mask[:, :, i:i+chunk_size, j:j+chunk_size]
                    attn_scores = attn_scores + chunk_mask
                
                # Store attention weights and context for later softmax
                chunk_attn_weights.append(attn_scores)
            
            # Concatenate all key chunks
            # [batch_size, num_heads, chunk_q, seq_len_k]
            attn_weights = torch.cat(chunk_attn_weights, dim=-1)
            
            # Apply softmax and dropout
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
            
            # Process key/value sequence in chunks again to compute final output
            for j in range(0, seq_len_k, chunk_size):
                v_chunk = v[:, :, j:j+chunk_size]
                attn_weights_chunk = attn_weights[:, :, :, j:j+chunk_size]
                
                # Compute weighted values for this chunk
                # [batch_size, num_heads, chunk_q, head_dim]
                context_chunk = torch.matmul(attn_weights_chunk, v_chunk)
                
                # Add to total context
                chunk_context.append(context_chunk)
            
            # Sum contributions from all key/value chunks
            # [batch_size, num_heads, chunk_q, head_dim]
            context = sum(chunk_context)
            
            # Update output
            output[:, :, i:i+chunk_size] = context
        
        return output
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False
    ) -> torch.Tensor:
        """
        Compute attention efficiently.
        
        Args:
            q: Query tensor of shape [batch_size, num_heads, seq_len_q, head_dim]
            k: Key tensor of shape [batch_size, num_heads, seq_len_k, head_dim]
            v: Value tensor of shape [batch_size, num_heads, seq_len_k, head_dim]
            mask: Attention mask of shape [batch_size, 1, seq_len_q, seq_len_k]
            causal: Whether to use causal masking
            
        Returns:
            Output tensor of shape [batch_size, num_heads, seq_len_q, head_dim]
        """
        # Use PyTorch's built-in flash attention if available and if we don't need a specific mask
        if self.use_builtin_flash_attn and (mask is None or causal):
            # Transpose inputs for PyTorch's built-in function
            # from [batch, num_heads, seq_len, head_dim] to [batch, seq_len, num_heads, head_dim]
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            
            # Call PyTorch's scaled_dot_product_attention with flash attention
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                attn_mask=None if causal else mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=causal
            )
            
            # Transpose back to original shape
            # from [batch, seq_len, num_heads, head_dim] to [batch, num_heads, seq_len, head_dim]
            return out.transpose(1, 2)
        
        # Fall back to chunked attention if built-in flash attention is not available
        # or if we have a specific mask that can't be represented as causal
        return self._chunk_attention(q, k, v, mask)


class FlashMHA(nn.Module):
    """
    Multi-head attention with Flash Attention.
    
    This implements efficient attention using a flash attention algorithm 
    when available, falling back to standard attention otherwise.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        """
        Initialize the Flash MHA module.
        
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(FlashMHA, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better training stability."""
        for module in [self.query, self.key, self.value, self.output]:
            nn.init.xavier_uniform_(module.weight)
    
    def to(self, *args, **kwargs):
        """Moves all model parameters to the specified device."""
        super().to(*args, **kwargs)
        return self
            
    def flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Compute attention using flash attention if available.
        
        Args:
            q: Query tensor of shape [batch_size, num_heads, seq_len_q, d_k]
            k: Key tensor of shape [batch_size, num_heads, seq_len_k, d_k]
            v: Value tensor of shape [batch_size, num_heads, seq_len_k, d_k]
            mask: Attention mask of shape [batch_size, 1, seq_len_q, seq_len_k]
            causal: Whether to use causal masking
            
        Returns:
            Output tensor of shape [batch_size, num_heads, seq_len_q, d_k]
        """
        # Check if FlashAttention is available
        if FLASH_ATTENTION_AVAILABLE:
            # Reshape tensors to match FlashAttention requirements
            q = q.transpose(1, 2)  # [batch_size, seq_len_q, num_heads, d_k]
            k = k.transpose(1, 2)  # [batch_size, seq_len_k, num_heads, d_k]
            v = v.transpose(1, 2)  # [batch_size, seq_len_k, num_heads, d_k]
            
            output = flash_attn_func(
                q, k, v,
                dropout_p=0.0,
                causal=causal,
                softmax_scale=1.0 / math.sqrt(self.d_k)
            )
            # Reshape output back
            output = output.transpose(1, 2)  # [batch_size, num_heads, seq_len_q, d_k]
            return output
        else:
            # Fall back to standard attention
            return self.standard_attention(q, k, v, mask, causal)
    
    def standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Compute standard scaled dot-product attention.
        
        Args:
            q: Query tensor of shape [batch_size, num_heads, seq_len_q, d_k]
            k: Key tensor of shape [batch_size, num_heads, seq_len_k, d_k]
            v: Value tensor of shape [batch_size, num_heads, seq_len_k, d_k]
            mask: Attention mask of shape [batch_size, seq_len]
            causal: Whether to use causal masking
            
        Returns:
            Output tensor of shape [batch_size, num_heads, seq_len_q, d_k]
        """
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask if requested
        if causal:
            seq_len = q.size(2)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            scores.masked_fill_(causal_mask, float("-inf"))
        
        # Apply attention mask if provided
        if mask is not None:
            # Convert mask to proper dimensions if needed
            # Original mask is [batch_size, seq_len]
            if mask.dim() == 2:
                # For encoder self-attention: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(2)
                scores.masked_fill_(mask == 0, float("-inf"))
            elif mask.dim() == 3:
                # For decoder cross-attention: [batch_size, tgt_len, src_len] -> [batch_size, 1, tgt_len, src_len]
                mask = mask.unsqueeze(1)
                scores.masked_fill_(mask == 0, float("-inf"))
            elif mask.dim() == 4:
                # Already in proper format [batch_size, heads, tgt_len, src_len]
                scores.masked_fill_(mask == 0, float("-inf"))
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores.float(), dim=-1).type_as(scores)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)
        
        return context
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into heads and transpose.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Reshaped tensor of shape [batch_size, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose to [batch_size, num_heads, seq_len, d_k]
        return x.transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine heads and transpose back.
        
        Args:
            x: Input tensor of shape [batch_size, num_heads, seq_len, d_k]
            
        Returns:
            Reshaped tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, _, seq_len, _ = x.size()
        # Transpose back to [batch_size, seq_len, num_heads, d_k]
        x = x.transpose(1, 2)
        # Combine heads
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_rotary: bool = False,
        rotary_emb: Optional[nn.Module] = None,
        position_ids: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of Flash MHA.
        
        Args:
            query: Query tensor of shape [batch_size, seq_len_q, d_model]
            key: Key tensor of shape [batch_size, seq_len_k, d_model]
            value: Value tensor of shape [batch_size, seq_len_k, d_model]
            mask: Attention mask of shape [batch_size, 1, seq_len_q, seq_len_k]
            use_rotary: Whether to use rotary embeddings
            rotary_emb: Rotary embedding module
            position_ids: Position IDs for rotary embeddings
            causal: Whether to use causal masking
            
        Returns:
            Output tensor of shape [batch_size, seq_len_q, d_model]
        """
        device = query.device
        batch_size = query.size(0)
        
        # Ensure all model parameters are on the correct device
        for param in self.parameters():
            if param.device != device:
                param.data = param.data.to(device)
        
        # Linear projections
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        
        # Split heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # Apply rotary embeddings if requested
        if use_rotary and rotary_emb is not None and position_ids is not None:
            q, k = rotary_emb(q, k, position_ids)
        
        # Flash attention
        context = self.flash_attention(q, k, v, mask, causal)
        
        # Combine heads
        context = self.combine_heads(context)
        
        # Final linear projection
        output = self.output(context)
        
        return output 