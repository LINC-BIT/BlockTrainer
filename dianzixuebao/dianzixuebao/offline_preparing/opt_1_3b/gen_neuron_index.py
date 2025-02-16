from utils.dl.common.env import set_random_seed
set_random_seed(1)

import sys
import torch
from torch import nn 
from utils.common.exp import get_res_save_dir, save_models_dict_for_init


if __name__ == '__main__':
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b')
    # encoded_input = tokenizer.encode_plus(
    #     'hello world', max_length=32, padding="max_length", truncation=True, return_tensors="pt"
    # )
    # x = {key: tensor.squeeze(0) for key, tensor in encoded_input.items()}
    # # print(x['input_ids'].size())
    # x['return_dict'] = False
    # print(x)
    # exit()
    # 1. Data
    from dianzixuebao.settings.pos_tagging import pos_tagging_scenario as scenario
    
    # 2. Model(s)
    fm_models_dict_path = save_models_dict_for_init(
        torch.load('dianzixuebao/offline_preparing/opt_1_3b/results/lora_fine_tune.py/20240515/999998-200637-new_loss/models/main_best_acc=0.7795.pt'),
        __file__, uid='fm'
    )
    import glob
    kb_models_dict_path = save_models_dict_for_init(
        torch.load('dianzixuebao/offline_preparing/opt_1_3b/results/gen_knowledge_base.py/20240515/999999-211258-new_loss/models/main_best_acc=0.7677.pt'),
        __file__, uid='kb'
    )
    from dianzixuebao.model_impl.opt.main import GenNeuronIndexModel_OPTTokenClassification
    fm_model = GenNeuronIndexModel_OPTTokenClassification(
        name='swin_v2_l_fm',
        models_dict_path=fm_models_dict_path,
        device='cuda'
    )
    kb_model = GenNeuronIndexModel_OPTTokenClassification(
        name='swin_v2_l_kb',
        models_dict_path=kb_models_dict_path,
        device='cuda'
    )
    fm_model.num_classes = scenario.num_classes
    kb_model.num_classes = scenario.num_classes
    
    # 3. Algorithm
    from methods.gen_neuron_index import GenNeuronIndexAlg
    alg = GenNeuronIndexAlg(
        models={
            'fm': fm_model,
            'kb': kb_model
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
                
            'example_sample': {'input_ids': torch.LongTensor([[    2, 42891,   232,     1,     1,     1,     1,     1,     1,     1,
            1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
            1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
            1,     1]]).to(device), 'attention_mask': torch.LongTensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0]]).to(device), 'return_dict': False},
            
            'qkv_layers_name': qkv_layers,
            'proj_layers_name': proj_layers,
            'ff1_layers_name': ff1_layers,
            'ff2_layers_name': ff2_layers,
            'window_merge': None,
            
            'fbs_r': 8,
            'min_sparsity': 0.,
            'max_sparsity': 0.9,
            'sparsity_loss_alpha': 1e-8,
            'neuron_index_loss_alpha': 1e-4,

            'train_batch_size': 8,
            'val_batch_size': 32,
            'num_workers': 16,
            'optimizer': 'AdamW',
            'optimizer_args': {'lr': 1e-4, 'momentum': 0.9, 'weight_decay': 1e-5},
            'neuron_index_optimizer': 'AdamW',
            'neuron_index_optimizer_args': {'lr': 3e-3, 'betas': [0.9, 0.999], 'weight_decay': 0.1},
            'scheduler': 'StepLR',
            'scheduler_args': {'step_size': 15000, 'gamma': 0.1},
            'num_iters': 40000,
            'val_freq': 1000,
            'only_gen_neuron_in_qkv': True,
            
            'fp16': False
        }
    )