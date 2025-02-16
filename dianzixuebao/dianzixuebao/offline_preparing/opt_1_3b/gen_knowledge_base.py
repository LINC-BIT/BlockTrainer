from utils.dl.common.env import set_random_seed
set_random_seed(1)

import sys
import torch
from torch import nn 
from utils.common.exp import get_res_save_dir, save_models_dict_for_init


if __name__ == '__main__':
    # 1. Data
    from dianzixuebao.settings.pos_tagging import pos_tagging_scenario as scenario
    
    # 2. Model(s)
    models_dict_path = 'dianzixuebao/offline_preparing/opt_1_3b/results/lora_fine_tune.py/20240515/999998-200637-new_loss/models/main_best_acc=0.7795.pt'
    
    from dianzixuebao.model_impl.opt.main import GenKnowledgeBaseModel_OPTTokenClassification
    model = GenKnowledgeBaseModel_OPTTokenClassification(
        name='opt',
        models_dict_path=models_dict_path,
        device='cuda'
    )
    model.num_classes = scenario.num_classes
    
    # 3. Algorithm
    from methods.gen_knowledge_base import GenKnowledgeBaseAlg
    alg = GenKnowledgeBaseAlg(
        models={
            'main': model
        },
        res_save_dir=get_res_save_dir(__file__, sys.argv[1])
    )
    
    # 4. Run!
    qkv_layers = [[f'model.decoder.layers.{i}.self_attn.q_proj', 
                   f'model.decoder.layers.{i}.self_attn.k_proj', 
                   f'model.decoder.layers.{i}.self_attn.v_proj'] for i in range(24)]
    proj_layers = [f'model.decoder.layers.{i}.self_attn.out_proj' for i in range(24)]
    ff1_layers = [f'model.decoder.layers.{i}.fc1' for i in range(24)]
    ff2_layers = [f'model.decoder.layers.{i}.fc2' for i in range(24)]
    
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
            'optimizer_args': {'lr': 1e-4, 'momentum': 0.9, 'weight_decay': 1e-4},
            'scheduler': 'StepLR',
            'scheduler_args': {'step_size': 15000, 'gamma': 0.1},
            'num_iters': 40000,
            'val_freq': 1000,
            
            'fp16': False
        }
    )