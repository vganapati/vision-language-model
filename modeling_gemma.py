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
            breakpoint()
            # key_cache = [layers, batch_size, num_heads_kv, seq_len, head_dim]
            return self.key_cache[0].shape[-2]
    
    def update(self,
               key_states: torch.Tensor, # [batch_size, num_heads_kv, seq_len, head_dim]
               value_states: torch.Tensor, # [batch_size, num_heads_kv, seq_len, head_dim]
               layer_idx: int,
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        breakpoint()
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
                 num_attention_heads, # number of heads for the queries
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
        self.weight = nn.Parameter(torch.zeros(dim)) # one for each feature (g_i) # hidden_dim
    
    def _norm(self, x):
        breakpoint()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # 1 / sqrt(...)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float()) # weight is initially 1
        return output.type_as(x)

class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, # head dim
                 max_position_embeddings=2048, base=10000, device=None):
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
    breakpoint()
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

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    breakpoint()
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads*n_rep, slen, head_dim)

class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads # for grouped query attention

        assert self.num_heads % self.num_key_value_heads == 0

        self.num_key_value_groups = self.num_heads // self.num_key_value_heads # for grouped query attention
        self.max_position_embeddings = config.max_position_embeddings # how many positions we can encode with RoPE
        self.rope_theta = config.rope_theta
        self.is_casual = True

        assert self.hidden_size % self.num_heads == 0

        breakpoint() # self.head_dim

        """
        Example Values:

        num_heads = 8
        hidden_size = 1024
        head_dim = 1024 / 8 = 128
        Wq is [1024, 8*128] = [1024, 1024]
        Wk is [1024, 2*128] = [1024, 256]
        Wv is [1024, 2*128] = [1024, 256]
        """

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads*self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads*self.head_dim, bias = config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads*self.head_dim, bias = config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads*self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                kv_cache: Optional[KVCache] = None,
                **kwargs,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        breakpoint()
        bsz, q_len, _ = hidden_states.size() # [batch_size, seq_len, hidden_size]

        # [batch_size, seq_len, num_heads_q * head_dim]
        query_states = self.q_proj(hidden_states)

        # [batch_size, seq_len, num_heads_kv * head_dim]
        key_states = self.k_proj(hidden_states)

        # [batch_size, seq_len, num_heads_kv * head_dim]
        value_states = self.v_proj(hidden_states)

        # [batch_size, num_heads_q, seq_len, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1,2)

        # [batch_size, num_heads_kv, seq_len, head_dim]
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)

        # [batch_size, num_heads_kv, seq_len, head_dim]
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)

        # [batch_size, seq_len, head_dim], [batch_size, seq_len, head_dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)

        breakpoint()
        # [batch_size, num_heads_q, seq_len, head_dim], [batch_size, num_heads_kv, seq_len, head_dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        breakpoint()
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)
        
        # repeat the key and values to match the number of heads of the query
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # perform the calculation as usual: Q @ K^T / sqrt(head_dim)
        # shape is [batch_size, num_heads_q, seq_len_q, seq_len_kv]
        attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) / math.sqrt(self.head_dim)

        breakpoint()
        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        # apply the softmax
        # [batch_size, num_heads_q, seq_len_q, seq_len_kv]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # apply the dropout
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # multiply by the values
        # [batch_size, num_heads_q, seq_len_q, seq_len_kv] x [batch_size, num_heads_kv, seq_len_kv, head_dim]
        # -> [batch_size, num_heads_q, seq_len_q, head_dim]
        attn_output = attn_weights @ value_states

        assert attn_output.size() == (bsz, self.num_heads, q_len, self.head_dim)

        # make the sequence length the second dimension
        # [batch_size, num_heads_q, seq_len_q, head_dim] -> [batch_size, seq_len_q, num_heads_q, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()

        # concatenate all the heads together
        # [batch_size, seq_len_q, num_heads_q, head_dim] -> [batch_size, seq_len_q, num_heads_q * head_dim]
        attn_output = attn_output.view(bsz, q_len, -1)

        # multiply by W_o -> [batch_size, seq_len_q, hidden_size]
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

class GemmaDecoderLayer(nn.Module):
    def __init__(self,
                 config: GemmaConfig,
                 layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        breakpoint() # layer_idx
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                kv_cache: Optional[KVCache] = None,
                ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        # [batch_size, seq_len, hidden_size]
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # [batch_size, seq_len, hidden_size]
        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        # [batch_size, seq_len, hidden_size]
        hidden_states = residual + hidden_states
        
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class GemmaModel(nn.Module):

    """
    Language Model
    """

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        breakpoint() # understand padding_idx
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        breakpoint()
        return self.embed_tokens
    
    def forward(self,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                kv_cache: Optional[KVCache] = None,
                ) -> torch.FloatTensor:
        
        # [batch_size, seq_len, hidden_size]
        hidden_states = inputs_embeds

        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            # [batch_size, seq_len, hidden_size]
            hidden_states = decoder_layer (
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )
        
        breakpoint()
        # [batch_size, seq_len, hidden_size]
        hidden_states = self.norm(hidden_states)

        # [batch_size, seq_len, hidden_size]
        return hidden_states

class GemmaForCausalLM(nn.Module):
    """
    Language model + Final linear layer
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                kv_cache: Optional[KVCache] = None,
                ) -> Tuple:
        
        # inputs_embeds = [batch_size, seq_len, hidden_size]
        # outputs = [batch_size, seq_len, hidden_size]

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            # return the updated cache
            return_data["kv_cache"] = kv_cache

        return return_data

class PaliGemmaMultiModalProjector(nn.Module):

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)
    
    def forward(self, image_features):
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, projection_dim]
        hidden_states = self.linear(image_features)
        return hidden_states

class PaliGemmaForConditionalGeneration(nn.Module):
    """
    Main structure
    """

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        self.language_model = GemmaForCausalLM(config.text_config)

        breakpoint()
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
    
    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def merge_input_ids_with_image_features(self,
                                            image_features: torch.Tensor,
                                            inputs_embeds: torch.Tensor, # embeddings of teokens
                                            input_ids: torch.Tensor, # numbers of tokens
                                            attention_mask: torch.Tensor,
                                            kv_cache: Optional[KVCache] = None,
                                            ):
        breakpoint()
        _, _, hidden_size = image_features.shape # XXX check this is hidden_size
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        # [batch_size, seq_len, hidden_size]
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        # combine the embeddings of the image tokens, the text tokens, and mask out all the padding tokens
        final_embedding = torch.zeros(batch_size, sequence_length, hidden_size,
                                      dtype=dtype, device=device)
        
        breakpoint()
        # [batch_size, seq_len]
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id) # True for text tokens
        image_mask = input_ids == self.config.image_token_index # True for image tokens
        pad_mask = input_ids == self.pad_token_id # True for padding tokens, don't have padding tokens in inference

        breakpoint()
        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, hidden_size)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, hidden_size)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, hidden_size)

        # add the text embeddings
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)

        breakpoint()
        # insert image embeddings
        # we can't use torch.where because the sequence length of scaled_image_features is not equal to the
        # sequence length of the final embeddings
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)

        # zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        ### CREATE THE ATTENTION MASK ###

        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        breakpoint()
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            breakpoint()
            # prefill step
            # do not mask any token, because we're in the prefill phase
            # this only works when we have no padding

            # causal_mask is -inf to mask a value out
            
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            breakpoint()
            # since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len

            # also in this case we don't need to mask anything, since each query should be able to attend
            # all previous tokens
            # this only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        breakpoint()
        # add the head dimension
        # [batch_size, q_len, kv_len] -> [batch_size, num_heads_q, q_len, kv_len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # the position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # create position_ids baed on the size of the attention mask
            # for masked tokens, use the number 1 as position
            breakpoint()
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0),1).to(device)
        
        breakpoint() # are all the contextualized embeddings used for next token prediction or only the last contextualized embedding
        return final_embedding, causal_mask, position_ids
    
    def forward(self,
                input_ids: torch.LongTensor = None,
                pixel_values: torch.FloatTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None,
                ) -> Tuple:
        
        breakpoint()

        # make sure the input is right-padded
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # 1. Extra the input embeddings
        # shape is [batch_size, seq_len, hidden_size]
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        # [batch_size, channels, height, width] -> [batch_size, num_patches, embed_dim]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))

        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, hidden_size]
        image_features = self.multi_modal_projector(selected_image_feature)

        # Merge the embeddings of the text tokens and the image tokens
        inputs_embeds, attention_mask, position_ids = self._merge_ids_with_image_features(image_features,
                                                                                          inputs_embeds,
                                                                                          input_ids,
                                                                                          attention_mask,
                                                                                          kv_cache,
                                                                                          )
        
        outputs = self.language_model(attention_mask=attention_mask,
                                      position_ids=position_ids,
                                      inputs_embeds=inputs_embeds,
                                      kv_cache=kv_cache,
                                      )
        
        return outputs
        





        
