from utils.dl.common.env import set_random_seed
set_random_seed(1)

import sys
import torch
from torch import nn 
from utils.common.exp import get_res_save_dir, save_models_dict_for_init


if __name__ == '__main__':
    # 1. Data
    from dianzixuebao.settings.seg import semantic_segmentation_scenario as scenario
    
    # 2. Model(s)
    fm_models_dict_path = save_models_dict_for_init(
        torch.load('dianzixuebao/offline_preparing/swinv2_l/seg/results/lora_fine_tune.py/20240509/999980-142931-trial/models/main_best_acc=0.3466.pt'),
        __file__, uid='fm'
    )
    import glob
    kb_models_dict_path = save_models_dict_for_init(
        torch.load('dianzixuebao/offline_preparing/swinv2_l/seg/results/gen_knowledge_base.py/20240509/999993-155739-trial/models/main_best_acc=0.3614.pt'),
        __file__, uid='kb'
    )
    from dianzixuebao.model_impl.swin_v2_l.main import GenNeuronIndexModel_SwinV2SemanticSegmentation
    fm_model = GenNeuronIndexModel_SwinV2SemanticSegmentation(
        name='swin_v2_l_fm',
        models_dict_path=fm_models_dict_path,
        device='cuda'
    )
    kb_model = GenNeuronIndexModel_SwinV2SemanticSegmentation(
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
    qkv_layers = []
    proj_layers = []
    ff1_layers = []
    ff2_layers = []
    window_merge = {}
    for si, stage in enumerate(kb_model.model.swinv2.encoder.layers):
        for bi, block in enumerate(stage.blocks):
            qkv_layers += [[f'swinv2.encoder.layers.{si}.blocks.{bi}.attention.self.query', 
                           f'swinv2.encoder.layers.{si}.blocks.{bi}.attention.self.key', 
                           f'swinv2.encoder.layers.{si}.blocks.{bi}.attention.self.value']]
            proj_layers += [f'swinv2.encoder.layers.{si}.blocks.{bi}.attention.output.dense']
            ff1_layers += [f'swinv2.encoder.layers.{si}.blocks.{bi}.intermediate.dense']
            ff2_layers += [f'swinv2.encoder.layers.{si}.blocks.{bi}.output.dense']
            
            for l in qkv_layers[-1]:
                window_merge[l + '.0'] = {0: 25, 1: 9, 2: 4, 3: 4}[si]
            window_merge[proj_layers[-1] + '.0'] = {0: 25, 1: 9, 2: 4, 3: 4}[si]
            window_merge[ff1_layers[-1]] = None
            window_merge[ff2_layers[-1]] = None
    
    alg.run(
        scenario=scenario, 
        hyps={
            'launch_tbboard': True,
                
            'example_sample': torch.rand((1, 3, 224, 224)),
            
            'qkv_layers_name': qkv_layers,
            'proj_layers_name': proj_layers,
            'ff1_layers_name': ff1_layers,
            'ff2_layers_name': ff2_layers,
            'window_merge': window_merge,
            
            'fbs_r': 8,
            'min_sparsity': 0.,
            'max_sparsity': 0.9,
            'sparsity_loss_alpha': 1e-8,
            'neuron_index_loss_alpha': 1e-4,

            'train_batch_size': 16,
            'val_batch_size': 128,
            'num_workers': 16,
            'optimizer': 'AdamW',
            'optimizer_args': {'lr': 3e-4, 'momentum': 0.9, 'weight_decay': 1e-5},
            'neuron_index_optimizer': 'AdamW',
            'neuron_index_optimizer_args': {'lr': 3e-3, 'betas': [0.9, 0.999], 'weight_decay': 0.1},
            'scheduler': 'StepLR',
            'scheduler_args': {'step_size': 15000, 'gamma': 0.1},
            'num_iters': 40000,
            'val_freq': 1000,
            
            'fp16': False
        }
    )