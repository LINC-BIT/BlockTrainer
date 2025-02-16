from utils.dl.common.env import set_random_seed
set_random_seed(1)

import sys
import torch
from torch import nn 
from utils.common.exp import get_res_save_dir, save_models_dict_for_init


if __name__ == '__main__':
    # 1. Data
    from dianzixuebao.settings.text_gen import scenario
    
    # 2. Model(s)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_3b_v2")
    # models_dict_path = save_models_dict_for_init({
    #     'main': model
    # }, __file__)
    from dianzixuebao.model_impl.llama.main import LoRAFineTuningModel_TextGeneration
    model = LoRAFineTuningModel_TextGeneration(
        name='opt',
        models_dict_path={'main': model},
        device='cuda'
    )
    
    # 3. Algorithm
    from methods.lora_ft import LoRATuningAlg
    alg = LoRATuningAlg(
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
    device = 'cuda'
    alg.run(
        scenario=scenario, 
        hyps={
            'launch_tbboard': True,
                
            'example_sample': {'input_ids': torch.tensor([[ 101, 5672, 2033, 2011, 2151, 3793, 2017, 1005, 1040, 2066, 1012,  102]]).to(device), 
                                  'token_type_ids': torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).to(device), 
                                  'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).to(device), 'return_dict': False},
            
            'qkv_layers_name': qkv_layers,
            'lora_r': 8,

            'train_batch_size': 1,
            'val_batch_size': 32,
            'num_workers': 8,
            'optimizer': 'AdamW',
            'optimizer_args': {'lr': 5e-5, 'momentum': 0.9, 'weight_decay': 1e-5},
            'scheduler': 'StepLR',
            'scheduler_args': {'step_size': 15000, 'gamma': 0.1},
            'num_iters': 40000,
            'val_freq': 1000,
            
            'fp16': True # 18GB vs 23GB
        }
    )