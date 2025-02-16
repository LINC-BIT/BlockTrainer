from transformers.models.vit.modeling_vit import prune_linear_layer
from transformers.pytorch_utils import find_pruneable_heads_and_indices
from utils.dl.common.model import set_module, LayerActivation2, get_module
import os
import torch
from torch import nn

# 2. OPT-1.3B, batch size 64
from transformers import OPTModel, OPTConfig
from transformers.models.opt.modeling_opt import OPTPreTrainedModel, SequenceClassifierOutputWithPast


class AttentionPrunableOPTModel(OPTModel):
    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.prune_heads_of_a_layer(self.decoder.layers[layer], heads)
        
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
        self.out_proj = prune_linear_layer(self.out_proj, index, dim=1)

        # Update hyper params and store pruned heads
        self.num_heads = self.num_heads - len(heads)
        self.embed_dim = self.head_dim * self.num_heads
        
        
class OPTForTokenClassification(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = AttentionPrunableOPTModel(config)
        self.classifier = nn.Linear(config.word_embed_proj_dim, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
    def _prune_heads(self, heads_to_prune):
        self.model._prune_heads(heads_to_prune)
        
    def set_num_classes(self, num_classes):
        self.classifier = nn.Linear(self.config.word_embed_proj_dim, num_classes, bias=False)
        self.post_init()
        
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        head_mask = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.classifier(hidden_states)

        # if input_ids is not None:
        #     batch_size, sequence_length = input_ids.shape[:2]
        # else:
        #     batch_size, sequence_length = inputs_embeds.shape[:2]

        # if self.config.pad_token_id is None:
        #     sequence_lengths = -1
        # else:
        #     if input_ids is not None:
        #         # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
        #         sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        #         sequence_lengths = sequence_lengths % input_ids.shape[-1]
        #         sequence_lengths = sequence_lengths.to(logits.device)
        #     else:
        #         sequence_lengths = -1
        #         logger.warning(
        #             f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
        #             "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
        #         )

        # pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            raise NotImplementedError
            
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )