import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

class KVCache():

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # key_cache = [batch_size, num_heads_kv, seq_len, head_dim]
            return self.key_cache[0].shape[-2]
    
    def update(self,
               key_states: torch.Tensor,
               value_states: torch.Tensor,
               layer_idx: int,
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # if we never added anything to the KV-cache for this layer, create it
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # ... otherwise we concatenate the new keys with the existing ones
            # each tensor has shape: [batch_size, num_heads_kv, seq_len, head_dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        
        # ... and then we return all the existing keys + the new ones
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

class GemmaConfig:
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 intermediate_size,
                 num_hidden_layers,
                 num_attention_heads,
                 num_key_value_heads,
                 head_dim=256,
                 max_position_embeddings=8192,
                 rms_norm_eps=1e-6,
                 rope_theta=10000.0,
                 attention_bias=False,
                 attention_dropout=0.0,
                 pad_token_id=None,
                 **kwargs,
                 ):
        breakpoint()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig:
    def __init__(self,
                 vision_config=None,
                 text_config=None,
                 ignore_index=-100,
                 image_token_index=256000,
                 vocab_size=257152,
                 projection_dim=2048,
                 hidden_size=2048,
                 pad_token_id=None,
                 **kwargs,
                 ):
        breakpoint()        
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        breakpoint()
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

class GemmaRMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim # it is set to the head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        breakpoint()
        # Calculcate the theta according to the formula theta_i = base^(-2i/dim) 
        # where i = 0, 1, 2, ..., dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x = [bs, num_attention_heads, seq_len, head_size]
        
        self.inv_freq.to(x.device)

        breakpoint()
        # copy the inv_freq tensor for batch in the sequence
        # inv_freq_expanded = [batch_size, head_dim//2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)

        # position_ids_expanded = [batch_size, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

        breakpoint()
        with torch.autocast(device_type=device_type, enabled=False):
            # multiply each theta by the position (which is the argument of the sin and cos functions)
            
            # freqs = [batch_size, head_dim // 2, 1] @ [batch_size, 1, seq_len] 
            # -> [batch_size, seq_len, head_dim // 2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

            # emb = [batch_size, seq_len, head_dim]
            emb = torch.cat((freqs, freqs), dim=-1)

            # cos, sin = [batch_size, seq_len, head_dim]
            cos = emb.cos()
            sin = emb.sin()

            breakpoint()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    # build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding
    breakpoint()
    x1 = x[..., : x.shape[-1] // 2] # takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :] # takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim) # add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim) # add the head dimension

    # apply the formula (34) of the rotary positional encoding paper
    q_embed = (q*cos) + (rotate_half(q) * sin)
    k_embed = (k*cos) + (rotate_half(k) * sin)

    return q_embed, k_embed

class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, x):
        """
        # equivalent to:
        y = self.gate_proj(x) # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, intermediate_size]
        y = torch.gelu(y, approximate="tanh")
        j = self.up_proj(x) # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, intermediate_size]
        z = y * j
        z = self.down_proj(x) # [batch_size, seq_len, intermediate_size] -> [batch_size, seq_len, hidden_size]
        """
        breakpoint()
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

def repeat_kv
