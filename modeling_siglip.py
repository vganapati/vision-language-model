from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    def __init__(
            self,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels=3,
            image_size=224,
            patch_size=16,
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            num_image_tokens: int = None,
            **kwargs,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens

breakpoint()

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # this indicates no padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1,-1)),
            persistent=False,
        )
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [batch_size, channels, height, width]

        # convolve the `patch_size` kernel over the image, 
        # with no overlapping patches since the stride is equal

        # the output of the convolution will have shape [batch_size, embed_dim, num_patches_h, num_patches_w]
        patch_embeds = self.patch_embedding(pixel_values)

        # [batch_size, embed_dim, num_patches_h, num_patches_w] -> [batch_size, embed_dim, num_patches]
        # num_patches = num_patches_h * num_patches_w
        embeddings = patch_embeds.flatten(2)

        # [batch_size, embed_dim, num_patches] -> [batch_size, num_patches, embed_dim]
        embeddings = embeddings.transpose(1,2)

        # add position embeddings to each patch. each positional encoding is a vector of size [embed_dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)

        # [batch_size, num_patches, embed_dim]
        return embeddings

class SiglipAttention(nn.Module):
    """Multi-headed attention from 'Attention is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads

        assert self.embed_dim % self.num_heads == 0, f"embed_dim {self.embed_dim} not divisible by num_heads {self.num_heads}"
        self.head_dim = self.embed_dim // self.num_heads

        self.scale = self.head_dim**(-0.5) # equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # hidden_states: [batch_size, num_patches, embed_dim]
        batch_size, seq_len, _ = hidden_states.size()

        # query_states: [batch_size, num_patches, embed_dim]
        query_states = self.q_proj(hidden_states)

        # key_states: [batch_size, num_patches, embed_dim]
        key_states = self.k_proj(hidden_states)

        # value_states: [batch_size, num_patches, embed_dim]
        value_states = self.v_proj(hidden_states)

        # [batch_size, num_patches, embed_dim] -> [batch_size, num_heads, num_patches, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        # calculate the attention using the formulat Q * K^T / sqrt(d_k)
        # attn_weights = [batch_size, num_heads, num_patches, num_patches]

        attn_weights = (query_states @ key_states.transpose(2,3))*self.scale
        assert attn_weights.size() == (batch_size, self.num_heads, seq_len, seq_len), f"attn weights should be size {(batch_size, self.num_heads, seq_len, seq_len)} not {attn_weights.size()}"

        # apply the softmax row-wise
        # attn_weights = [batch_size, num_heads, num_patches, num_patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # multiply the attention weights by the value states
        # attn_output = [batch_size, num_heads, num_patches, head_dim]
        attn_output = attn_weights @ value_states

        assert attn_output.size() == (batch_size, self.num_heads, seq_len, self.head_dim)

        # [batch_size, num_heads, num_patches, head_dim] -> [batch_size, num_patches, num_heads, head_dim]
        attn_output = attn_output.transpose(1,2).contiguous()

        # [batch_size, num_patches, num_heads, head_dim] -> [batch_size, num_patches, embed_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        # final projection layer
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, self.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)

        # nonlinearity
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")

        # [batch_size, num_patches, intermediate_size] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        # residual = [batch_size, num_patches, embed_dim]
        residual = hidden_states

        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm1(hidden_states)

        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states, _ = self.self_attn(hidden_states)

        # [batch_size, num_patches, embed_dim]
        hidden_states = residual + hidden_states

        # residual = [batch_size, num_patches, embed_dim]
        residual = hidden_states

        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm2(hidden_states)

        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.mlp(hidden_states)

        # [batch_size, num_patches, embed_dim]
        hidden_states = residual + hidden_states

        return hidden_states

class SiglipEncoder(nn.Module):

        
