from logging import getLogger
import torch.nn as nn
from ._base import *
from ..utils.import_utils import compare_transformers_version

if compare_transformers_version("v4.28.0", op="ge"):
    from ..nn_modules.fused_llama_attn import FusedLlamaAttentionForQuantizedModel
    from ..nn_modules.fused_llama_mlp import FusedLlamaMLPForQuantizedModel
else:
    FusedLlamaAttentionForQuantizedModel = None
    FusedLlamaMLPForQuantizedModel = None

logger = getLogger(__name__)


class TransnormerGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "TransnormerDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens"]
    inside_layer_modules = [
        ["channel_mixer.l1", "channel_mixer.l2", "channel_mixer.l3"],
        ["token_mixer.qkvu_proj"],
        ["token_mixer.out_proj"],

    ]


    fused_attn_module_type = FusedLlamaAttentionForQuantizedModel
    fused_mlp_module_type = FusedLlamaMLPForQuantizedModel


__all__ = ["TransnormerGPTQForCausalLM"]