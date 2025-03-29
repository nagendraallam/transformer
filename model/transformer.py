"""
Transformer model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple
from .rotary_embeddings import RotaryEmbedding
from .rmsnorm import RMSNorm
try:
    from .flash_attention import FlashMHA
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization as used in modern transformer designs.
    More efficient and stable than LayerNorm in many scenarios.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) - more effective for handling 
    position information than traditional positional encoding.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build cosine and sine cache
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            dtype=torch.get_default_dtype()
        )
        
    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        
    def forward(self, x, seq_len=None):
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype)
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Applies Rotary Position Embeddings to q and k tensors."""
    # Get embeddings from the global cos/sin cache
    if position_ids is not None:
        # Handle invalid indices
        max_pos = cos.shape[0] - 1
        position_ids = torch.clamp(position_ids, 0, max_pos)
        
        # Gather the correct embeddings for each position
        cos = cos[position_ids]
        sin = sin[position_ids]
    
    # Reshape to match the expected shape for rotary embeddings
    if cos.dim() == 3 and q.dim() == 4:  # (batch, seq, dim) vs (batch, heads, seq, dim)
        cos = cos.unsqueeze(1)  # (batch, 1, seq, dim)
        sin = sin.unsqueeze(1)  # (batch, 1, seq, dim)
    
    # Apply rotary embeddings
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V, and output
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        
        self.softmax_scale = self.d_k ** -0.5  # Scale factor for attention
        
    def split_heads(self, x):
        # Reshape to (batch_size, seq_len, num_heads, d_k)
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose to (batch_size, num_heads, seq_len, d_k)
        return x.transpose(1, 2)
    
    def forward(self, query, key, value, mask=None, position_ids=None, use_rotary=False, rotary_emb=None):
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor of shape [batch_size, seq_len_q, d_model]
            key: Key tensor of shape [batch_size, seq_len_k, d_model]
            value: Value tensor of shape [batch_size, seq_len_k, d_model]
            mask: Attention mask of shape [batch_size, seq_len_q, seq_len_k]
            position_ids: Position IDs for rotary embeddings
            use_rotary: Whether to use rotary embeddings
            rotary_emb: Rotary embedding module
            
        Returns:
            Output tensor of shape [batch_size, seq_len_q, d_model]
        """
        batch_size = query.size(0)
        
        # Linear projections
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        
        # Split into heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # Apply rotary embeddings if enabled
        if use_rotary and rotary_emb is not None:
            # Get the cached cos and sin from the rotary embeddings module
            if hasattr(rotary_emb, 'cos_cached') and hasattr(rotary_emb, 'sin_cached'):
                cos = rotary_emb.cos_cached
                sin = rotary_emb.sin_cached
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
            else:
                # Fall back to the module's forward method
                seq_length = q.shape[2]
                if position_ids is None:
                    position_ids = torch.arange(seq_length, device=q.device)
                    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                q, k = rotary_emb(q, k, position_ids)
        
        # Scaled dot-product attention
        # q: [batch_size, num_heads, seq_len_q, d_k]
        # k: [batch_size, num_heads, seq_len_k, d_k]
        # v: [batch_size, num_heads, seq_len_k, d_k]
        
        # Calculate attention scores
        # [batch_size, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.softmax_scale
        
        # Apply attention mask if provided
        if mask is not None:
            # Prepare mask for proper broadcasting
            if mask.dim() == 2:
                # Convert from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # Convert from [batch_size, seq_len_q, seq_len_k] to [batch_size, 1, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(1)
            
            # Use a smaller constant for float16 compatibility
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask == 0, min_value)
        
        # Apply softmax to get attention weights - cast to fp32 for stability
        attention_weights = torch.softmax(scores.float(), dim=-1).type_as(scores)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.output(context)  # [batch_size, seq_len_q, d_model]
        
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, activation="gelu", dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # GELU activation as used in modern transformer models
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation="gelu", use_flash_attention=False):
        super(EncoderLayer, self).__init__()
        
        # Use Flash MHA if requested and available
        if use_flash_attention and FLASH_ATTENTION_AVAILABLE:
            self.self_attention = FlashMHA(d_model, num_heads, dropout)
        else:
            self.self_attention = MultiHeadAttention(d_model, num_heads)
            
        self.feed_forward = FeedForward(d_model, d_ff, activation, dropout)
        
        # Use RMSNorm instead of LayerNorm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, position_ids=None, use_rotary=False, rotary_emb=None, causal=False):
        # Pre-LN architecture: Apply normalization before attention
        normalized_x = self.norm1(x)
        
        if isinstance(self.self_attention, FlashMHA):
            attn_output = self.self_attention(
                normalized_x, normalized_x, normalized_x, 
                mask=mask, use_rotary=use_rotary, rotary_emb=rotary_emb, 
                position_ids=position_ids, causal=causal
            )
        else:
            attn_output = self.self_attention(
                normalized_x, normalized_x, normalized_x, 
                mask=mask, position_ids=position_ids, use_rotary=use_rotary, rotary_emb=rotary_emb
            )
        
        # Residual connection
        x = x + self.dropout(attn_output)
        
        # Pre-LN architecture for feed forward
        normalized_x = self.norm2(x)
        ff_output = self.feed_forward(normalized_x)
        
        # Residual connection
        x = x + self.dropout(ff_output)
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation="gelu", use_flash_attention=False):
        super(DecoderLayer, self).__init__()
        
        # Use Flash MHA if requested and available
        if use_flash_attention and FLASH_ATTENTION_AVAILABLE:
            self.self_attention = FlashMHA(d_model, num_heads, dropout)
            self.cross_attention = FlashMHA(d_model, num_heads, dropout)
        else:
            self.self_attention = MultiHeadAttention(d_model, num_heads)
            self.cross_attention = MultiHeadAttention(d_model, num_heads)
            
        self.feed_forward = FeedForward(d_model, d_ff, activation, dropout)
        
        # Use RMSNorm instead of LayerNorm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None, position_ids=None, use_rotary=False, rotary_emb=None):
        # Pre-LN architecture: Apply normalization before attention
        normalized_x = self.norm1(x)
        
        if isinstance(self.self_attention, FlashMHA):
            attn_output = self.self_attention(
                normalized_x, normalized_x, normalized_x, 
                mask=tgt_mask, use_rotary=use_rotary, rotary_emb=rotary_emb, 
                position_ids=position_ids, causal=True
            )
        else:
            attn_output = self.self_attention(
                normalized_x, normalized_x, normalized_x, 
                tgt_mask, position_ids, use_rotary, rotary_emb
            )
        
        # Residual connection
        x = x + self.dropout(attn_output)
        
        # Cross attention with pre-normalization
        normalized_x = self.norm2(x)
        
        if isinstance(self.cross_attention, FlashMHA):
            attn_output = self.cross_attention(
                normalized_x, encoder_output, encoder_output, 
                mask=src_mask
            )
        else:
            attn_output = self.cross_attention(
                normalized_x, encoder_output, encoder_output, 
                src_mask
            )
        
        # Residual connection
        x = x + self.dropout(attn_output)
        
        # Feed forward with pre-normalization
        normalized_x = self.norm3(x)
        ff_output = self.feed_forward(normalized_x)
        
        # Residual connection
        x = x + self.dropout(ff_output)
        
        return x


class Transformer(nn.Module):
    def __init__(
        self, 
        src_vocab_size, 
        tgt_vocab_size, 
        d_model=512, 
        num_heads=8, 
        num_encoder_layers=6,
        num_decoder_layers=6, 
        d_ff=2048, 
        max_seq_length=2048, 
        dropout=0.1,
        activation="gelu",
        use_rotary_embeddings=True
    ):
        super(Transformer, self).__init__()
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Option to use either traditional or rotary positional embeddings
        self.use_rotary_embeddings = use_rotary_embeddings
        
        if use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(
                dim=d_model // num_heads, 
                max_position_embeddings=max_seq_length
            )
        else:
            self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
            
        # Flash attention support
        self.flash_attention_enabled = False
            
        # Create encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model, 
                num_heads, 
                d_ff, 
                dropout, 
                activation,
                use_flash_attention=False  # Will be updated in forward pass
            )
            for _ in range(num_encoder_layers)
        ])
        
        # Create decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                d_model, 
                num_heads, 
                d_ff, 
                dropout, 
                activation,
                use_flash_attention=False  # Will be updated in forward pass
            ) 
            for _ in range(num_decoder_layers)
        ])
        
        # Final normalization layer
        self.final_norm = RMSNorm(d_model)
        
        self.final_layer = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
    def configure_optimizations(self, flash_attention: bool = False, gradient_checkpointing: bool = False):
        """
        Configure model optimizations.
        
        Args:
            flash_attention: Whether to use flash attention
            gradient_checkpointing: Whether to use gradient checkpointing
        """
        self.flash_attention_enabled = flash_attention and FLASH_ATTENTION_AVAILABLE
        
        if self.flash_attention_enabled:
            # Re-create the attention layers to use flash attention
            for layer in self.encoder_layers:
                layer.self_attention = FlashMHA(
                    d_model=self.d_model,
                    num_heads=layer.self_attention.num_heads,
                    dropout=layer.dropout.p,
                )
            
            for layer in self.decoder_layers:
                layer.self_attention = FlashMHA(
                    d_model=self.d_model,
                    num_heads=layer.self_attention.num_heads,
                    dropout=layer.dropout.p,
                )
                layer.cross_attention = FlashMHA(
                    d_model=self.d_model,
                    num_heads=layer.cross_attention.num_heads,
                    dropout=layer.dropout.p,
                )
        
        # Setup gradient checkpointing
        if gradient_checkpointing:
            self.gradient_checkpointing_enable()
        
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        # Wrap the encoder layers with gradient checkpointing
        for layer in self.encoder_layers:
            # Create a custom forward function that preserves function signature
            def custom_encoder_forward(layer_module, *inputs, **kwargs):
                def closure(*inputs, **kwargs):
                    return layer_module(*inputs, **kwargs)
                return closure
            
            # Store original forward method
            orig_forward = layer.forward
            
            # Define a new forward method for checkpointing
            def make_checkpointed_forward(module, orig_forward):
                def checkpointed_forward(x, *args, **kwargs):
                    # Use checkpoint with the original function signature
                    return torch.utils.checkpoint.checkpoint(
                        orig_forward,
                        x, *args,
                        use_reentrant=False,
                        **kwargs
                    )
                return checkpointed_forward
            
            # Apply the checkpointed forward
            layer.forward = make_checkpointed_forward(layer, orig_forward)
        
        # Apply gradient checkpointing to decoder layers
        for layer in self.decoder_layers:
            # Store original forward method
            orig_forward = layer.forward
            
            # Define a new forward method for checkpointing
            def make_checkpointed_forward(module, orig_forward):
                def checkpointed_forward(x, encoder_output, *args, **kwargs):
                    # Use checkpoint with the original function signature
                    return torch.utils.checkpoint.checkpoint(
                        orig_forward,
                        x, encoder_output, *args,
                        use_reentrant=False,
                        **kwargs
                    )
                return checkpointed_forward
            
            # Apply the checkpointed forward
            layer.forward = make_checkpointed_forward(layer, orig_forward)
        
    def generate_square_subsequent_mask(self, size):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        This mask ensures that during self-attention, a token cannot attend to subsequent tokens.
        """
        # Create a square matrix where subsequent positions are masked (upper triangle)
        mask = (1 - torch.triu(torch.ones(size, size), diagonal=1)).bool()
        return mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass for transformer with enhanced features.
        
        Args:
            src: Source tensor of shape [batch_size, src_seq_len]
            tgt: Target tensor of shape [batch_size, tgt_seq_len]
            src_mask: Source mask of shape [batch_size, src_seq_len] (optional)
                      Used to mask padding tokens in the encoder
            tgt_mask: Target mask of shape [batch_size, tgt_seq_len] (optional)
                      Used for masked self-attention in the decoder
                      
        Returns:
            Output tensor of shape [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # Get sequence lengths and batch size, ensuring correct dimensions
        if src.dim() > 2:
            # If src has more than 2 dimensions, reshape it
            src_shape = src.size()
            src = src.view(-1, src_shape[-1])
            if src_mask is not None:
                src_mask = src_mask.view(-1, src_mask.size(-1))
                
        if tgt.dim() > 2:
            # If tgt has more than 2 dimensions, reshape it
            tgt_shape = tgt.size()
            tgt = tgt.view(-1, tgt_shape[-1])
            if tgt_mask is not None:
                tgt_mask = tgt_mask.view(-1, tgt_mask.size(-1))
        
        # Get batch size and sequence lengths
        batch_size, src_seq_len = src.size()
        _, tgt_seq_len = tgt.size()
        
        # Validate input to prevent out-of-bounds errors
        max_seq_length = getattr(self, 'rotary_emb', None).max_position_embeddings if self.use_rotary_embeddings else 5000
        
        # Create position ids for rotary embeddings if needed
        if self.use_rotary_embeddings:
            # Ensure position ids are within bounds
            src_position_ids = torch.arange(min(src_seq_len, max_seq_length), device=src.device).unsqueeze(0).expand(batch_size, -1)
            tgt_position_ids = torch.arange(min(tgt_seq_len, max_seq_length), device=tgt.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed source tokens
        src_embedded = self.encoder_embedding(src) * math.sqrt(self.d_model)
        
        # Apply positional encoding if not using rotary embeddings
        if not self.use_rotary_embeddings:
            src_embedded = self.positional_encoding(src_embedded)
        
        src_embedded = self.dropout(src_embedded)
        
        # Embed target tokens
        tgt_embedded = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        
        # Apply positional encoding if not using rotary embeddings
        if not self.use_rotary_embeddings:
            tgt_embedded = self.positional_encoding(tgt_embedded)
        
        tgt_embedded = self.dropout(tgt_embedded)
        
        # Handle source mask (for padding)
        if src_mask is None:
            # If no source mask is provided, assume all tokens are valid
            src_padding_mask = torch.ones(batch_size, src_seq_len, dtype=torch.bool, device=src.device)
        else:
            # Use the provided source mask
            src_padding_mask = src_mask
        
        # Handle target mask (for causal attention)
        # For decoder self-attention, we need to combine:
        # 1. Causal mask to prevent attending to future tokens
        # 2. Padding mask (if provided in tgt_mask)
        
        # Create causal mask
        causal_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
        
        # If target mask is provided (as a padding mask), combine with causal mask
        if tgt_mask is not None:
            # Convert padding mask [batch_size, tgt_seq_len] to proper format for combination
            # with causal mask [tgt_seq_len, tgt_seq_len]
            tgt_padding_mask = tgt_mask.unsqueeze(1).expand(batch_size, tgt_seq_len, tgt_seq_len)
            # Combine causal and padding masks
            # Both must be True for a position to be attended to
            combined_mask = causal_mask.unsqueeze(0) & tgt_padding_mask
        else:
            # Just use the causal mask expanded to batch size
            combined_mask = causal_mask.unsqueeze(0).expand(batch_size, tgt_seq_len, tgt_seq_len)
        
        # Pass through encoder layers
        encoder_output = src_embedded
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(
                encoder_output, 
                src_padding_mask,
                src_position_ids if self.use_rotary_embeddings else None,
                self.use_rotary_embeddings,
                self.rotary_emb if self.use_rotary_embeddings else None,
                causal=False
            )
            
        # Pass through decoder layers
        decoder_output = tgt_embedded
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(
                decoder_output,
                encoder_output,
                src_mask=src_padding_mask,
                tgt_mask=combined_mask,
                position_ids=tgt_position_ids if self.use_rotary_embeddings else None,
                use_rotary=self.use_rotary_embeddings,
                rotary_emb=self.rotary_emb if self.use_rotary_embeddings else None
            )
        
        # Apply final normalization
        decoder_output = self.final_norm(decoder_output)
            
        # Final output projection
        output = self.final_layer(decoder_output)
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions.
    Implementation adapted from PyTorch's tutorial.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize the positional encoding.
        
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create a buffer to store the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and store
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for positional encoding.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        # Add positional encoding to the input
        # The [:, :x.size(1), :] ensures we only use the first x.size(1) positions
        return x + self.pe[:, :x.size(1), :] 