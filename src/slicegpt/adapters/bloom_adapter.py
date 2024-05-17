from typing import cast, Optional, Tuple, Union

import torch
from torch import FloatTensor, LongTensor, Tensor, matmul
from torch.nn import Linear, Module, LayerNorm
from transformers import PretrainedConfig, PreTrainedTokenizerBase
from transformers.models.bloom.modeling_bloom import BloomConfig, BloomBlock, BloomForCausalLM

from slicegpt.model_adapter import LayerAdapter, ModelAdapter

class CompressBloomBlock(BloomBlock):
    """
    This class simulates the BloomBlock class from transformers
    but with the addition of a shortcut_Q attribute. This attribute is used to rotate the residual tensors.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        residual = hidden_states

        layernorm_output = self.input_layernorm(hidden_states)

        # Layer norm post the self attention.
        # but `apply_residual_connection_post_layernorm` is false, so we don't do it.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP.
        output = self.mlp(layernorm_output, residual)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions
    
class BloomBlockAdapter(LayerAdapter):
    def __init__(self, layer: BloomBlock) -> None:
        super().__init__()
        self._layer: BloomBlock = layer

    @property
    def layer(self) -> Module:
        return self._layer
    
    @property
    def hidden_states_args_position(self) -> int:
        return 0

    @property
    def hidden_states_output_position(self) -> int:
        return 0

    def get_first_layernorm(self) -> Module:
        return self.layer.input_layernorm

    def get_second_layernorm(self) -> Module:
        return self.layer.post_attention_layernorm
    
    def get_attention_inputs(self) -> list[Linear]:
        return [self.layer.self_attention.query_key_value]
    
    def get_attention_output(self) -> Linear:
        return self.layer.self_attention.dense
    
    def get_mlp_inputs(self) -> list[Linear]:
        return [self.layer.mlp.dense_h_to_4h]

    def get_mlp_output(self) -> Linear:
        return self.layer.mlp.dense_4h_to_h

class BloomModelAdapter(ModelAdapter):
    def __init__(self, model: BloomForCausalLM) -> None:
        super().__init__()
        self._model: BloomForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config
    
    @property
    def config_type(self) -> type:
        return BloomConfig
    
    @property
    def parallel_blocks(self) -> bool:
        return False

    @property
    def seqlen(self) -> int:
        # TODO: Maybe wrong!
        return self.config.max_position_embeddings
    
    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def should_bake_mean_into_linear(self) -> bool:
        return False
    
    @property
    def original_layer_type(self) -> type:
        return BloomBlock

    @property
    def original_layer_norm_type(self) -> type:
        return LayerNorm

    @property
    def layer_adapter_type(self) -> type:
        return BloomBlockAdapter

    @property
    def compressed_layer_type(self) -> type:
        return CompressBloomBlock
    
    @property
    def use_cache(self) -> bool:
        return self.config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self.config.use_cache = value

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        return self.model(input_ids=input_ids).logits

    def convert_layer_to_compressed(self, layer: Module, layer_idx: int | None) -> Module:
        compressed_layer = self.compressed_layer_type(cast(self.config_type, self.config)).to(
            self.config.torch_dtype
        )
        compressed_layer.load_state_dict(layer.state_dict(), strict=True)
        return compressed_layer

    def get_layers(self) -> list[LayerAdapter]:
        return [self.layer_adapter_type(layer) for layer in self.model.transformer.h]

    def get_raw_layer_at(self, index: int) -> Module:
        return self.model.transformer.h[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.transformer.h[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self.model.transformer.word_embeddings]

    def get_pre_head_layernorm(self) -> type:
        pre_head_layernorm = self.model.transformer.word_embeddings_layernorm
        assert isinstance(pre_head_layernorm, self.original_layer_norm_type)
        return pre_head_layernorm

    def get_lm_head(self) -> Linear:
        return self.model.lm_head

    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # Llama-2 doesn't have a pad token by default
        tokenizer.pad_token = tokenizer.eos_token
        self.config.pad_token_id = tokenizer.pad_token_id

    @classmethod
    def _from_pretrained(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if not model_name.startswith("bigscience/bloom-7b1"):
            return None

        model = BloomForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model.config.torch_dtype = dtype

        return BloomModelAdapter(model)

    @classmethod
    def _from_uninitialized(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if not model_name.startswith("bigscience/bloom-7b1"):
            return None

        class UninitializedBloomForCausalLM(BloomForCausalLM):
            def _init_weights(self, _) -> None:
                # Prevent weight initialization
                pass

        config = BloomConfig.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model = UninitializedBloomForCausalLM(config)
        model = model.to(dtype=dtype)

        return BloomModelAdapter(model)
