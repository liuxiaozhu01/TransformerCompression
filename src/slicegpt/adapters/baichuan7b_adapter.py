from typing import cast, List, Optional, Tuple, Union

import torch
from torch import FloatTensor, LongTensor, Tensor, matmul
from torch.nn import Linear, Module
from transformers import PretrainedConfig, PreTrainedTokenizerBase
from slicegpt.models.hf_baichuan.baichuan7B.modeling_baichuan_7B import RMSNorm, DecoderLayer, BaiChuanForCausalLM
from slicegpt.models.hf_baichuan.baichuan7B.configuration_baichuan import BaiChuanConfig

from slicegpt.model_adapter import LayerAdapter, ModelAdapter

class CompressedBaichuan7BDecoderLayer(DecoderLayer):
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            build_dp: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            build_dp=build_dp,
        )
        if self.attn_shortcut_Q is not None:
            rotated_residual = matmul(residual, self.attn_shortcut_Q)
            hidden_states = rotated_residual + hidden_states
        else:
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.mlp_shortcut_Q is not None:
            rotated_residual = matmul(residual, self.mlp_shortcut_Q)
            hidden_states = rotated_residual + hidden_states
        else:
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
class Baichuan7BLayerAdapter(LayerAdapter):
    def __init__(self, layer: DecoderLayer) -> None:
        super().__init__()
        self._layer: DecoderLayer = layer

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
        return [self.layer.self_attn.W_pack]
    
    def get_attention_output(self) -> Linear:
        return self.layer.self_attn.o_proj
    
    def get_mlp_inputs(self) -> list[Linear]:
        return [self.layer.mlp.gate_proj, self.layer.mlp.up_proj]

    def get_mlp_output(self) -> Linear:
        return self.layer.mlp.down_proj
    
class Baichuan7BModelAdapter(ModelAdapter):
    def __init__(self, model: BaiChuanForCausalLM) -> None:
        super().__init__()
        self._model: BaiChuanForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config
    
    @property
    def config_type(self) -> type:
        return BaiChuanConfig
    
    @property
    def parallel_blocks(self) -> bool:
        return False

    @property
    def seqlen(self) -> int:
        return self.config.max_position_embeddings

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def should_bake_mean_into_linear(self) -> bool:
        return False
    
    @property
    def original_layer_type(self) -> type:
        return DecoderLayer

    @property
    def original_layer_norm_type(self) -> type:
        return RMSNorm

    @property
    def layer_adapter_type(self) -> type:
        return Baichuan7BLayerAdapter

    @property
    def compressed_layer_type(self) -> type:
        return CompressedBaichuan7BDecoderLayer
    
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
        return [self.layer_adapter_type(layer) for layer in self.model.model.layers]

    def get_raw_layer_at(self, index: int) -> Module:
        return self.model.model.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.model.layers[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self.model.model.embed_tokens]

    def get_pre_head_layernorm(self) -> type:
        pre_head_layernorm = self.model.model.norm
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
        
        if not model_name.startswith("baichuan-inc/Baichuan2-7B-Base"):
            return None
        
        model = BaiChuanForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model.config.torch_dtype = dtype
        
        return Baichuan7BModelAdapter(model)
    
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
        if not model_name.startswith("baichuan-inc/Baichuan2-7B-Base"):
            return None

        class UninitializedBaiChuanForCausalLM(BaiChuanForCausalLM):
            def _init_weights(self, _) -> None:
                # Prevent weight initialization
                pass

        config = BaiChuanConfig.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model = UninitializedBaiChuanForCausalLM(config)
        model = model.to(dtype=dtype)

        return Baichuan7BModelAdapter(model)