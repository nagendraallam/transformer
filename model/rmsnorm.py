"""
Root Mean Square Layer Normalization.
Implementation of RMSNorm from the paper:
"Root Mean Square Layer Normalization" (https://arxiv.org/abs/1910.07467)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    This is a more computationally efficient alternative to LayerNorm.
    """
    
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Initialize the RMSNorm layer.
        
        Args:
            dim: Hidden dimension
            eps: Small constant for numerical stability
        """
        super(RMSNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.
        
        Args:
            x: Input tensor of shape [..., dim]
            
        Returns:
            Normalized tensor of same shape as input
        """
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        x_normalized = x / rms
        
        # Apply learnable scale
        return self.scale * x_normalized 