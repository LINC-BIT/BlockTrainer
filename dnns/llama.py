from transformers import ViTForImageClassification, ViTConfig
from transformers.models.vit.modeling_vit import prune_linear_layer
from transformers.pytorch_utils import find_pruneable_heads_and_indices
from utils.dl.common.model import set_module, LayerActivation2, get_module
import os
import torch
import tqdm
from transformers import ViTForImageClassification, BertForSequenceClassification, AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig


class AttentionPrunableLlamaForCausalLM(LlamaForCausalLM):
    def prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.prune_heads_of_a_layer(self.model.layers[layer], heads)
        
    def prune_heads_of_a_layer(self, layer, heads) -> None:
        self = layer.self_attn
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_heads, self.head_dim, set()
        )

        # Prune linear layers
        self.q_proj = prune_linear_layer(self.q_proj, index)
        self.k_proj = prune_linear_layer(self.k_proj, index)
        self.v_proj = prune_linear_layer(self.v_proj, index)
        self.o_proj = prune_linear_layer(self.o_proj, index, dim=1)

        # Update hyper params and store pruned heads
        self.num_heads = self.num_heads - len(heads)
        self.num_key_value_heads = self.num_key_value_heads - len(heads)
        self.hidden_size = self.num_heads * self.head_dim
        
        
def get_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('openlm-research/open_llama_3b_v2', padding_side='left')
    # tokenizer = AutoTokenizer.from_pretrained(os.environ['model_path'], padding_side='left')
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.sep_token = tokenizer.eos_token
    special_tokens = {"pad_token":"<pad>"}#, "sep_token":"<|sep|>", "bos_token":"<|bos|>"}
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = "<pad>"
    # tokenizer.bos_token = "<|bos|>"
    # tokenizer.sep_token = "<|sep|>"
    return tokenizer
