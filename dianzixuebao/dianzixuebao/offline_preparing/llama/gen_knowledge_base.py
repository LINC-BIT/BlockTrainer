from utils.dl.common.env import set_random_seed
set_random_seed(1)

import sys
import torch
from torch import nn 
from utils.common.exp import get_res_save_dir, save_models_dict_for_init


def collate_fn(batch):
    dict = {}
    input_ids = []
    attention_mask = []
    token_type_ids = []
    labels = []
    return_dict = True
    lenli = []
    for item in batch:
        if len(item) == 1 or len(item['labels']) == 0:
            continue
        input_ids.append(item['input_ids'].unsqueeze(0))
        if 'attention_mask' in item.keys():
            attention_mask.append(item['attention_mask'].unsqueeze(0))
        if 'token_type_ids' in item.keys():
            token_type_ids.append(item['token_type_ids'].unsqueeze(0))
        labels.append(item['labels'].unsqueeze(0))
        if 'len' in item.keys():
            lenli.append(item['len'])

    dict['return_dict'] = batch[0]['return_dict']
    if len(input_ids) > 0:
        dict['input_ids'] = torch.cat(input_ids, dim=0)
    else:
        return {}, torch.Tensor([0])
    if len(attention_mask) > 0:
        dict['attention_mask'] = torch.cat(attention_mask, dim=0)
    if len(token_type_ids) > 0:
        dict['token_type_ids'] = torch.cat(token_type_ids, dim=0)
    dict['labels'] = torch.cat(labels, dim=0)
    if len(lenli) > 0:
        dict['len'] = lenli
    return dict, torch.Tensor([0])


if __name__ == '__main__':
    # 1. Data
    from dianzixuebao.settings.text_gen import scenario
    
    # 2. Model(s)
    # from transformers import AutoModelForCausalLM
    from dnns.llama import AttentionPrunableLlamaForCausalLM, get_tokenizer
    tokenizer = get_tokenizer()
    model = AttentionPrunableLlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b_v2")
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.tie_weights()
    
    from dianzixuebao.model_impl.llama.main import GenKnowledgeBaseModel_TextGeneration
    model = GenKnowledgeBaseModel_TextGeneration(
        name='opt',
        models_dict_path={'main': model},
        device='cuda'
    )
    
    # 3. Algorithm
    from methods.gen_knowledge_base import GenKnowledgeBaseAlg
    alg = GenKnowledgeBaseAlg(
        models={
            'main': model
        },
        res_save_dir=get_res_save_dir(__file__, sys.argv[1])
    )
    
    # 4. Run!
    qkv_layers = [[f'model.layers.{i}.self_attn.q_proj', 
                   f'model.layers.{i}.self_attn.k_proj', 
                   f'model.layers.{i}.self_attn.v_proj'] 
                  for i in range(26)]
    proj_layers = [f'model.layers.{i}.self_attn.o_proj' for i in range(24)]
    ff1_layers = [[f'model.layers.{i}.mlp.gate_proj', f'model.layers.{i}.mlp.up_proj'] for i in range(26)]
    ff2_layers = [f'model.layers.{i}.mlp.down_proj' for i in range(26)]
    
    device = 'cuda'
    alg.run(
        scenario=scenario, 
        hyps={
            'launch_tbboard': True,
                
            'example_sample': {'input_ids': torch.tensor([[ 101, 5672, 2033, 2011, 2151, 3793, 2017, 1005, 1040, 2066, 1012,  102]]).to(device), 
                                  'token_type_ids': torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).to(device), 
                                  'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).to(device), 'return_dict': False},
            
            'qkv_layers_name': qkv_layers,
            'proj_layers_name': proj_layers,
            'ff1_layers_name': ff1_layers,
            'ff2_layers_name': ff2_layers,
            'reducing_width_ratio': 0.875,

            'train_batch_size': 8,
            'val_batch_size': 32,
            'num_workers': 16,
            'optimizer': 'AdamW',
            'optimizer_args': {'lr': 5e-5, 'momentum': 0.9, 'weight_decay': 1e-4},
            'scheduler': 'StepLR',
            'scheduler_args': {'step_size': 15000, 'gamma': 0.1},
            'num_iters': 40000,
            'val_freq': 1000,
            
            'fp16': True,
            'collate_fn': collate_fn
        }
    )