"""
Implementation for Whisper architecture.
TODO: add docstring
"""

import dataclasses
import logging

from typing import Any, Dict, Optional, Tuple

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_chat.support import logging
from mlc_chat.support.config import ConfigBase
from mlc_chat.support.style import bold

import math

logger = logging.getLogger(__name__)

"""
HuggingFace's implementation of Whisper:https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py

"""


@dataclasses.dataclass
class WhisperConfig(ConfigBase):
    vocab_size: int
    num_mel_bins: int
    encoder_layers: int
    encoder_attention_heads: int
    decoder_layers: int
    decoder_attention_heads: int
    decoder_ffn_dim: int
    encoder_ffn_dim: int
    d_model: int
    max_source_positions: int
    max_target_positions: int
    pad_token_id: int
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.d_model % self.decoder_attention_heads != 0:
            raise ValueError(
                f"d_model must be divisible by decoder_attention_heads (got `d_model`: {self.d_model}"
                f" and `decoder_attention_heads`: {self.decoder_attention_heads})."
            )
        if self.d_model % self.encoder_attention_heads != 0:
            raise ValueError(
                f"d_model must be divisible by encoder_attention_heads (got `d_model`: {self.d_model}"
                f" and `encoder_attention_heads`: {self.encoder_attention_heads})."
            )
        if self.context_window_size == 0:
            for name in ["n_positions", "max_sequence_length", "max_target_positions"]:
                if name in self.kwargs or hasattr(self, name):
                    self.context_window_size = (
                        self.kwargs.pop(name) if name in self.kwargs else getattr(self, name)
                    )
                    logger.info(
                        "%s not found in config.json. Falling back to %s (%d)",
                        bold("context_window_size"),
                        bold(name),
                        self.context_window_size,
                    )
                    break
            else:
                raise ValueError(
                    "Unable to determine the maxmimum sequence length, because none of "
                    "`context_window_size`, `n_positions`, `max_sequence_length` or `max_target_positions` is "
                    "provided in `config.json`."
                )

            if self.prefill_chunk_size == 0:
                # chunk size same as context window size by default
                self.prefill_chunk_size = self.context_window_size


class WhisperPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.weight = nn.Parameter((max_seq_len, embed_dim))

    def forward(self, x: Tensor, offset: tir.Var):
        def te_op(x: te.Tensor, embed: te.Tensor, offset: tir.Var):
            def compute(i: tir.Var, j: tir.Var, k: tir.Var):
                return embed[offset + j, k]

            return te.compute([*x.shape, embed.shape[-1]], compute, name="position_embedding")

        pos_embed = nn.tensor_expr_op(te_op, "position_embedding", args=[x, self.weight, offset])
        return pos_embed


class WhisperAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, kv_cache_len: int, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        if kv_cache_len > 0:
            self.k_cache = nn.KVCache(kv_cache_len, [self.num_heads, self.head_dim])
            self.v_cache = nn.KVCache(kv_cache_len, [self.num_heads, self.head_dim])

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        cached_cross_attn_states: Optional[Tuple[Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        total_seq_len: Optional[tir.Var] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor]]]:
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states and cached_cross_attn_states).

        Args:

            hidden_states: hidden states of the encoder or decoder. Shape: [bsz, seq_len, embed_dim]
            key_value_states: hidden states of the encoder. Shape: [bsz, seq_len, embed_dim]
            cached_cross_attn_states: cached key and value states of the encoder. Tuple of two tensors(k and v) and each of shape [bsz, seq_len, num_heads, head_dim]
            attention_mask: attention mask. Shape: [bsz, seq_len, seq_len]
            total_seq_len: total length of the sequence. Used for caching the key and value states of the encoder.
        """

        is_cross_attention = key_value_states is not None or cached_cross_attn_states is not None

        h, d = self.num_heads, self.head_dim
        bsz, q_len, _ = hidden_states.shape
        assert bsz == 1, "Only support batch size 1 at this moment."

        q = nn.reshape(self.q_proj(hidden_states) * self.scaling, (bsz, q_len, h, d))

        dtype = q.dtype

        # initialize the cached_kv to 0
        def _initialize(q: Tensor):
            bsz, q_len, h, d = q.shape
            return te.compute([bsz, q_len, h, d], lambda i, j, k, l: 0)

        cached_kv = (
            op.tensor_expr_op(_initialize, name_hint="k", args=[q]),
            op.tensor_expr_op(_initialize, name_hint="v", args=[q]),
        )

        if is_cross_attention:
            # cross attention
            if cached_cross_attn_states is None:
                # no cache, cross attentions
                kv_len = key_value_states.shape[1]

                # Need to change the dtype of key_value_states to that for the quantizations
                key_value_states = key_value_states.astype(dtype)
                k = nn.reshape(self.k_proj(key_value_states), (bsz, kv_len, h, d))
                v = nn.reshape(self.v_proj(key_value_states), (bsz, kv_len, h, d))
                cached_kv = (k, v)

            else:
                # reuse cached k,v, cross_attentions
                k, v = cached_cross_attn_states

                # Need to chnage the dtype for the quantization
                k = k.astype(dtype)
                v = v.astype(dtype)

        else:
            # self attention
            k = nn.reshape(self.k_proj(hidden_states), (bsz, q_len, h, d))
            v = nn.reshape(self.v_proj(hidden_states), (bsz, q_len, h, d))

            if total_seq_len is not None:
                # reuse cached k, v, self_attention
                self.k_cache.append(nn.squeeze(k, axis=0))
                self.v_cache.append(nn.squeeze(v, axis=0))
                k = nn.reshape(self.k_cache.view(total_seq_len), (bsz, total_seq_len, h, d))
                v = nn.reshape(self.v_cache.view(total_seq_len), (bsz, total_seq_len, h, d))
            else:
                # encode self attention, no cache
                # self attention
                ...

        q = nn.permute_dims(q, [0, 2, 1, 3])  # [b, h, q_len, d]
        k = nn.permute_dims(k, [0, 2, 1, 3])  # [b, h, q_len, d]
        v = nn.permute_dims(v, [0, 2, 1, 3])  # [b, h, q_len, d]

        attn_weights = nn.matmul(q, (nn.permute_dims(k, [0, 1, 3, 2])))  # [b, h, q_len, q_len]

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        dtype = attn_weights.dtype
        attn_weights = attn_weights.maximum(tir.min_value(dtype))
        attn_weights = attn_weights.minimum(tir.max_value(dtype))
        if dtype == "float32":
            attn_weights = nn.softmax(attn_weights, axis=-1)
        else:
            attn_weights = nn.softmax(attn_weights.astype("float32"), axis=-1).astype(dtype)
        attn_output = nn.matmul(attn_weights, v)  # [b, h, q_len, d]

        attn_output = nn.permute_dims(attn_output, [0, 2, 1, 3])  # [b, q_len, h, d]
        attn_output = nn.reshape(attn_output, (bsz, q_len, self.embed_dim))  # [b, q_len, h * d]

        attn_output = self.out_proj(attn_output)
        # op.print_(attn_output)

        """
        if we have past_key_value, but not cached_cross_attn_states then
        attn_output is a tuple of two tensors (attn_output, cached_kv)
        so that we can use cached_kv for the next step of decoding as 
        the cached_cross_attn_states
        """
        """
        if is_cross_attention and cached_cross_attn_states is None:
            return attn_output, cached_kv
        else:
            return attn_output, None
        """
        return attn_output, cached_kv


class EncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            kv_cache_len=0,  # no need for kv_cache
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            key_value_states=None,
            cached_cross_attn_states=None,
            attention_mask=attention_mask,
            total_seq_len=None,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = nn.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = hidden_states.maximum(tir.min_value(hidden_states.dtype)).minimum(
            hidden_states
        )

        return hidden_states


class DecoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            kv_cache_len=config.max_target_positions,  # cache for self attention
        )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = WhisperAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            kv_cache_len=config.max_target_positions,  # cache for cross attention
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        cached_encoder_hidden_states: Tensor,
        total_seq_len: tir.Var,
        attention_mask: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor]]]:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            total_seq_len=total_seq_len,
            key_value_states=None,
            cached_cross_attn_states=None,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states, cross_attn_key_value = self.encoder_attn(
            hidden_states=hidden_states,
            total_seq_len=total_seq_len,
            key_value_states=encoder_hidden_states,
            cached_cross_attn_states=cached_encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = nn.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if cached_encoder_hidden_states is None:
            return hidden_states, cross_attn_key_value
        else:
            return hidden_states, None


class WhisperEncoder(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions

        self.conv1 = nn.Conv1D(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1D(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)

        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, input_features: Tensor) -> Tensor:
        expected_seq_length = self.max_source_positions * self.conv1.stride * self.conv2.stride
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        # for quantization purposes
        input_features = input_features.astype(self.conv1.weight.dtype)

        inputs_embeds = nn.gelu(self.conv1(input_features))
        inputs_embeds = nn.gelu(self.conv2(inputs_embeds))

        inputs_embeds = nn.permute_dims(inputs_embeds, [0, 2, 1])

        # embed_pos = self.embed_positions.weight
        # Position Embeddings
        # Generate np.arange(0, input_embeds.shape[1])
        def _input_positions(inputs: te.Tensor):
            b, s, _ = inputs.shape
            offset = 0
            return te.compute(
                (b, s), lambda _, j: (offset + j).astype("int32"), name="input_positions"
            )

        input_positions = op.tensor_expr_op(
            _input_positions,
            name_hint="input_positions",
            args=[inputs_embeds],
        )

        embed_pos = self.embed_positions(input_positions)

        hidden_states = inputs_embeds + embed_pos

        for _, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states)

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class WhisperDecoder(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()

        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_positions = WhisperPositionalEmbedding(self.max_target_positions, config.d_model)

        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_layers)])

        self.layer_norm = nn.LayerNorm(
            config.d_model,
        )

    def forward(
        self,
        input_ids: Tensor,
        total_seq_len: Optional[tir.Var] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        cached_encoder_key_value: Optional[Tuple[Tuple[Tensor]]] = None,
        attention_mask: Optional[Tensor] = None,
    ):
        # total_seq_len = Length of generated tokens
        input_embeds = self.embed_tokens(input_ids)
        past_seq_len = total_seq_len - 1
        position_embeds = self.embed_positions(input_ids, offset=past_seq_len)

        hidden_states = input_embeds + position_embeds
        # hidden_states = input_embeds

        all_encoder_key_value = ()
        for idx, decoder_layer in enumerate(self.layers):
            ith_cached_encoder_key_value = (
                cached_encoder_key_value[idx] if cached_encoder_key_value is not None else None
            )
            hidden_states, encoder_key_value = decoder_layer(
                hidden_states=hidden_states,
                total_seq_len=total_seq_len,
                encoder_hidden_states=encoder_hidden_states,
                cached_encoder_hidden_states=ith_cached_encoder_key_value,
                attention_mask=attention_mask,
            )
            if cached_encoder_key_value is None:
                all_encoder_key_value += (encoder_key_value,)

        hidden_states = self.layer_norm(hidden_states)

        if cached_encoder_key_value is None:
            return hidden_states, all_encoder_key_value
        else:
            return hidden_states, None


class WhisperModel(nn.Module):
    def __init__(self, config: WhisperConfig):
        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)


class WhisperForConditionalGeneration(nn.Module):
    def __init__(self, config: WhisperConfig):
        self.config = config
        self.model = WhisperModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor, total_seq_len: Optional[tir.Var] = None) -> Tensor:
        hidden_states, _ = self.model(input_ids=input_ids, total_seq_len=total_seq_len)

        def _index(x: te.Tensor):
            """
            x[:-1,:]. Extract the last hidden state of the sequence for each batch (x[i, seq_len - 1, k]).
            The shape (bsz, 1, d_embed) suggests that it reshapes the tensor to keep only the final
            hidden state for each item in the batch.
            """
            bsz, seq_len, d_embed = x.shape
            return te.compute(
                (bsz, 1, d_embed),
                lambda i, _, k: x[i, seq_len - 1, k],
                name="index",
            )

        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.proj_out(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def encode(self, input_ids: Tensor) -> Tensor:
        return self.model.encoder(input_ids)

    def softmax_with_temperature(self, logits: Tensor, temperature: Tensor):
        """Softmax."""
        return op.softmax(logits / temperature, axis=-1)

    def decode(
        self, input_ids: Tensor, total_seq_len: int, encoder_hidden_states: Tensor
    ) -> Tuple[Tensor, Tuple[Tuple[Tensor]]]:
        hidden_states, all_encoder_key_value = self.model.decoder.forward(
            input_ids=input_ids,
            total_seq_len=total_seq_len,
            encoder_hidden_states=encoder_hidden_states,
            cached_encoder_key_value=None,
            attention_mask=None,
        )
        lm_logits = self.proj_out(hidden_states)
        return lm_logits, all_encoder_key_value

    def prefill(
        self, input_ids: Tensor, total_seq_len: int, cached_encoder_key_value: Tuple[Tuple[Tensor]]
    ) -> Tensor:
        hidden_states, _ = self.model.decoder.forward(
            input_ids=input_ids,
            total_seq_len=total_seq_len,
            encoder_hidden_states=None,
            cached_encoder_key_value=cached_encoder_key_value,
            attention_mask=None,
        )
        lm_logits = self.proj_out(hidden_states)
        return lm_logits

    def get_default_spec(self):
        """Needed for ``export_tvm()``."""
        batch_size = 1
        encode_input_ndim = 16000 * 30 // 160

        mod_spec = {
            "encode": {
                "input_ids": nn.spec.Tensor(
                    [batch_size, self.config.num_mel_bins, encode_input_ndim], "float32"
                ),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "decode": {
                "input_ids": nn.spec.Tensor([batch_size, 1], "int32"),
                "total_seq_len": int,
                "encoder_hidden_states": nn.spec.Tensor(
                    [batch_size, self.config.max_source_positions, self.config.d_model], "float32"
                ),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "prefill": {
                "input_ids": nn.spec.Tensor([batch_size, "seq_len"], "int32"),
                "total_seq_len": int,
                "cached_encoder_key_value": tuple(
                    tuple(
                        nn.spec.Tensor(
                            [
                                1,
                                self.config.max_source_positions,
                                self.config.decoder_attention_heads,
                                self.config.d_model // self.config.decoder_attention_heads,
                            ],
                            "float32",
                        )
                        for i2 in range(2)
                    )
                    for i1 in range(self.config.encoder_layers)
                ),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
