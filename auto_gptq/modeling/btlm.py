from ._base import *


class BTLMGPTQLMHeadModel(BaseGPTQForCausalLM):
    layer_type = "BTLMBlock"
    layers_block_name = "transformer.h"
    outside_layer_modules = ["transformer.wte", "transformer.ln_f"]
    inside_layer_modules = [
        ["attn.c_attn"],
        ["attn.c_proj"],
        ["mlp.c_fc", "mlp.c_fc2"],
        ["mlp.c_proj"]
    ]


__all__ = ["BTLMGPTQLMHeadModel"]