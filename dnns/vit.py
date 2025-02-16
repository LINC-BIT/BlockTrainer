from transformers import ViTForImageClassification, ViTImageProcessor
from transformers.models.vit.modeling_vit import ViTSelfAttention, prune_linear_layer
from transformers.pytorch_utils import find_pruneable_heads_and_indices
import torch
from torch import nn 
from PIL import Image
import requests


class SoftmaxIgnoringZero(nn.Module):
    def __init__(self):
        super(SoftmaxIgnoringZero, self).__init__()
    
    def forward(self, x: torch.Tensor):
        # non_zero_x_indexes = x.nonzero(as_tuple=True)[0]
        # non_zero_x = x[non_zero_x_indexes]
        # non_zero_x_softmax = F.softmax(non_zero_x, self.dim, _stacklevel=5)
        # res = torch.zeros_like(x)

        # original: e^i / \sum_i e^i
        # ignoring zero: e^i
        # print(x)
        
        non_zero_mask = x != 0
        
        if non_zero_mask.sum() == x.numel():
            return F.softmax(x, -1)
        
        t = non_zero_mask.sum(-1)
        # assert t.view(-1).unique().size(0) == 1, f'{t.view(-1).unique()}, {x.size()}' # all vectors in the softmaxed dim has the same number of 0
        # assert t.view(-1).unique().size(0) <= 2, f'{t.view(-1).unique()}, {x.size()}' # all vectors in the softmaxed dim has the same number of 0 or has no 0
        non_zero_x = torch.masked_select(x, non_zero_mask)
        
        non_zero_x = non_zero_x.view(*(list(x.size())[0: -1] + [t.view(-1)[0].item()]))
        
        # print(non_zero_x)
        
        non_zero_x_softmax = F.softmax(non_zero_x, -1)
        
        a = x.nonzero(as_tuple=True)[-1]
        a = a.view(*non_zero_x_softmax.size())
        x = x.scatter(x.dim() - 1, a, non_zero_x_softmax)
        
        return x


url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
processor = ViTImageProcessor.from_pretrained('dnns/ckpts/vit-b-p16-224')
inputs = processor(images=image, return_tensors="pt")


model1 = ViTForImageClassification.from_pretrained('dnns/ckpts/vit-b-p16-224')
model2 = ViTForImageClassification.from_pretrained('dnns/ckpts/vit-b-p16-224')
model1.eval()
model2.eval()

for n, m in model1.named_modules():
    


@torch.no_grad()
def simulate_prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0):
    index = index.to(layer.weight.device)
    if dim == 0:
        layer.weight[index] = 0. 
        layer.bias[index] = 0.
    elif dim == 1:
        layer.weight[:, index] = 0.
    return layer
    

def simulate_prune_head(model, pruned_heads):
    random_masks = {}
    
    for block_index, block in enumerate(model.vit.encoder.layer):
        attention = block.attention
        heads = pruned_heads[block_index]

        heads, index = find_pruneable_heads_and_indices(
            heads, attention.attention.num_attention_heads, attention.attention.attention_head_size, attention.pruned_heads
        )

        # Prune linear layers
        attention.attention.query = simulate_prune_linear_layer(attention.attention.query, index)
        attention.attention.key = simulate_prune_linear_layer(attention.attention.key, index)
        attention.attention.value = simulate_prune_linear_layer(attention.attention.value, index)
        attention.output.dense = simulate_prune_linear_layer(attention.output.dense, index, dim=1)
        

pruned_heads = {
    h: torch.randperm(12)[0: 2].sort()[0]
    for h in range(12)
}

simulate_prune_head(model1, pruned_heads)
model2.prune_heads(pruned_heads)

o1 = model1(**inputs).logits
o2 = model2(**inputs).logits

print(((o1 - o2) ** 2).sum())