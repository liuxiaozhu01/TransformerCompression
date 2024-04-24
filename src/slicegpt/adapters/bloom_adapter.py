from typing import cast, Optional, Tuple, Union

import torch
from torch import FloatTensor, LongTensor, Tensor, matmul
from torch.nn import Linear, Module
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
        # if self.apply_residual_connection_post_layernorm:
        #     residual = layernorm_output
        # else:
        #     residual = hidden_states

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