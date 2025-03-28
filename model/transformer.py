import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but should be saved and loaded with the model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to the input
        x = x + self.pe[:, :x.size(1)]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V, and output
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        
    def split_heads(self, x):
        # Reshape to (batch_size, seq_len, num_heads, d_k)
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose to (batch_size, num_heads, seq_len, d_k)
        return x.transpose(1, 2)
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor of shape [batch_size, seq_len_q, d_model]
            key: Key tensor of shape [batch_size, seq_len_k, d_model]
            value: Value tensor of shape [batch_size, seq_len_v, d_model]
            mask: Optional mask tensor for masked attention
                If 2D: shape [batch_size, seq_len] - attention mask for padding
                If 3D: shape [batch_size, seq_len_q, seq_len_k] - custom attention pattern
                If 4D: shape [batch_size, num_heads, seq_len_q, seq_len_k] - per-head mask
        
        Returns:
            Output tensor of shape [batch_size, seq_len_q, d_model]
        """
        batch_size = query.size(0)
        
        # Linear projections
        query = self.query(query)  # [batch_size, seq_len_q, d_model]
        key = self.key(key)        # [batch_size, seq_len_k, d_model]
        value = self.value(value)  # [batch_size, seq_len_v, d_model]
        
        # Split heads
        query = self.split_heads(query)  # [batch_size, num_heads, seq_len_q, d_k]
        key = self.split_heads(key)      # [batch_size, num_heads, seq_len_k, d_k]
        value = self.split_heads(value)  # [batch_size, num_heads, seq_len_v, d_k]
        
        # Scaled dot-product attention
        # [batch_size, num_heads, seq_len_q, d_k] × [batch_size, num_heads, d_k, seq_len_k]
        # = [batch_size, num_heads, seq_len_q, seq_len_k]
        key_transpose = key.transpose(-1, -2)
        scores = torch.matmul(query, key_transpose) / math.sqrt(self.d_k)
        
        # Apply attention mask if provided
        if mask is not None:
            # Prepare mask for proper broadcasting based on its dimensions
            if mask.dim() == 2:
                # Convert from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
                # This is typically a padding mask
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # Convert from [batch_size, seq_len_q, seq_len_k] to [batch_size, 1, seq_len_q, seq_len_k]
                # This can be a causal mask
                mask = mask.unsqueeze(1)
            
            # Apply mask
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        # [batch_size, num_heads, seq_len_q, seq_len_k] × [batch_size, num_heads, seq_len_v, d_k]
        # = [batch_size, num_heads, seq_len_q, d_k]
        context = torch.matmul(attention_weights, value)
        
        # Concatenate heads and put through final linear layer
        # [batch_size, num_heads, seq_len_q, d_k] -> [batch_size, seq_len_q, num_heads * d_k]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.output(context)  # [batch_size, seq_len_q, d_model]
        
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self attention with residual connection and normalization
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection and normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self attention with residual connection and normalization
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross attention with residual connection and normalization
        attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed forward with residual connection and normalization
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 d_model=512, 
                 num_heads=8, 
                 num_encoder_layers=6,
                 num_decoder_layers=6, 
                 d_ff=2048, 
                 max_seq_length=100, 
                 dropout=0.1):
        super(Transformer, self).__init__()
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Create encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_encoder_layers)
        ])
        
        # Create decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_decoder_layers)
        ])
        
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
    def generate_square_subsequent_mask(self, size):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        This mask ensures that during self-attention, a token cannot attend to subsequent tokens.
        """
        # Create a square matrix where subsequent positions are masked
        mask = (1 - torch.triu(torch.ones(size, size), diagonal=1)).bool()
        return mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass for transformer.
        
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
        # Get sequence lengths and batch size
        batch_size, src_seq_len = src.size()
        _, tgt_seq_len = tgt.size()
        
        # Embed and add positional encoding for source
        src_embedded = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.positional_encoding(src_embedded)
        src_embedded = self.dropout(src_embedded)
        
        # Embed and add positional encoding for target
        tgt_embedded = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
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
            encoder_output = encoder_layer(encoder_output, src_padding_mask)
            
        # Pass through decoder layers
        decoder_output = tgt_embedded
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(
                decoder_output,
                encoder_output,
                src_mask=src_padding_mask,
                tgt_mask=combined_mask
            )
            
        # Final output projection
        output = self.final_layer(decoder_output)
        return output 