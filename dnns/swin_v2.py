from einops import rearrange
from transformers import Swinv2Model, Swinv2PreTrainedModel
from transformers.models.swinv2.modeling_swinv2 import Swinv2ImageClassifierOutput
import numpy as np
from torch import nn 
from transformers.pytorch_utils import find_pruneable_heads_and_indices
from transformers.models.vit.modeling_vit import prune_linear_layer
import torch
import torch.nn.functional as F


class AttentionPrunableSwinv2(Swinv2Model):
    def prune_heads(self, heads_to_prune):
        depths = self.config.depths
        cum_depths = list(np.cumsum(depths))
        for layer_index, heads in heads_to_prune.items():

            stage_index = 0
            while True:
                if layer_index < cum_depths[stage_index]:
                    break
                stage_index += 1
            block_index = layer_index - (cum_depths[stage_index - 1] if stage_index > 0 else 0)
                
            # self.swinv2.encoder.layers[stage_index].blocks[block_index].attention.prune_heads(heads)
            self.prune_heads_in_a_layer(self.swinv2.encoder.layers[stage_index].blocks[block_index].attention, heads)
            
    def prune_heads_in_a_layer(self, layer, heads):
        self = layer
        raw_heads = heads[:]
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.continuous_position_bias_mlp[2] = prune_linear_layer(self.self.continuous_position_bias_mlp[2], 
                                                                       torch.LongTensor([i for i in range(self.self.num_attention_heads) if i not in raw_heads]))

        
        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((self.self.num_attention_heads, 1, 1))))
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)
        
        

# Copied from transformers.models.swin.modeling_swin.SwinForImageClassification with SWIN->SWINV2,Swin->Swinv2,swin->swinv2
class Swinv2ForSeg(Swinv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.swinv2 = AttentionPrunableSwinv2(config)

        # Classifier head
        self.classifier = (
            DecoderLinear(21, 32, self.swinv2.num_features, (224, 224))
        )

        # Initialize weights and apply final processing
        self.post_init()
        
    def prune_heads(self, heads_to_prune):
        depths = self.config.depths
        cum_depths = list(np.cumsum(depths))
        for layer_index, heads in heads_to_prune.items():

            stage_index = 0
            while True:
                if layer_index < cum_depths[stage_index]:
                    break
                stage_index += 1
            block_index = layer_index - (cum_depths[stage_index - 1] if stage_index > 0 else 0)
                
            # self.swinv2.encoder.layers[stage_index].blocks[block_index].attention.prune_heads(heads)
            self.prune_heads_in_a_layer(self.swinv2.encoder.layers[stage_index].blocks[block_index].attention, heads)
            
    def prune_heads_in_a_layer(self, layer, heads):
        self = layer
        raw_heads = heads[:]
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.continuous_position_bias_mlp[2] = prune_linear_layer(self.self.continuous_position_bias_mlp[2], 
                                                                       torch.LongTensor([i for i in range(self.self.num_attention_heads) if i not in raw_heads]))

        
        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((self.self.num_attention_heads, 1, 1))))
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        pixel_values,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.swinv2(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # pooled_output = outputs[1]
        # print(outputs)
        # print([oi.size() for oi in outputs])

        logits = self.classifier(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            raise NotImplementedError
            
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return Swinv2ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states,
        )

        
        
class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder, im_size):
        super(DecoderLinear, self).__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls
        self.im_size = im_size

        self.head = nn.Linear(self.d_encoder, n_cls)

    def forward(self, x):
        # print('inside debug')
        # self.debug()
        # x = x[:, 1:] # remove cls token
        # print(x.size())
        
        H, W = self.im_size
        GS = H // self.patch_size
        # print(H, W, GS, self.patch_size)
        # print('head', self.head.weight.size(), x.size())
        # print(self.head, 'debug()')
        # print(x.size())
        x = self.head(x)
        
        # print(x.size())
        # print(x.size())
        
        # (b, HW//ps**2, ps_c)
        x = rearrange(x, "b (h w) c -> b c h w", h=GS)
        
        # print(x.size())
        
        masks = x
        masks = F.upsample(masks, size=(H, W), mode="bilinear")
        
        # print(masks.size())

        return masks