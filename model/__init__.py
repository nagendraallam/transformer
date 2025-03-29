from .transformer import Transformer, MultiHeadAttention, EncoderLayer, DecoderLayer
from .gpt2_model import PreTrainedGPT2
from .rmsnorm import RMSNorm
from .rotary_embeddings import RotaryEmbedding

try:
    from .flash_attention import FlashAttention, FlashMHA
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False 