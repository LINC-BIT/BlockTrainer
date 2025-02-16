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
    from dnns.swin_v2 import Swinv2ForSeg
    model = Swinv2ForSeg.from_pretrained('microsoft/swinv2-large-patch4-window12-192-22k')
    models_dict_path = save_models_dict_for_init({
        'main': model
    }, __file__)
    from dianzixuebao.model_impl.swin_v2_l.main import LoRAFineTuningModel_SwinV2SemanticSegmentation
    model = LoRAFineTuningModel_SwinV2SemanticSegmentation(
        name='swin_v2_l',
        models_dict_path=models_dict_path,
        device='cuda'
    )
    model.num_classes = scenario.num_classes
    
    # 3. Algorithm
    from methods.lora_ft import LoRATuningAlg
    alg = LoRATuningAlg(
        models={
            'main': model
        },
        res_save_dir=get_res_save_dir(__file__, sys.argv[1])
    )
    
    # 4. Run!
        
    qkv_layers = []
    for si, stage in enumerate(model.model.swinv2.encoder.layers):
        for bi, block in enumerate(stage.blocks):
            qkv_layers += [[f'swinv2.encoder.layers.{si}.blocks.{bi}.attention.self.query', 
                           f'swinv2.encoder.layers.{si}.blocks.{bi}.attention.self.key', 
                           f'swinv2.encoder.layers.{si}.blocks.{bi}.attention.self.value']]
            
    alg.run(
        scenario=scenario, 
        hyps={
            'launch_tbboard': True,
                
            'example_sample': torch.rand((1, 3, 224, 224)),
            
            'qkv_layers_name': qkv_layers,
            'lora_r': 8,

            'train_batch_size': 16,
            'val_batch_size': 128,
            'num_workers': 8,
            'optimizer': 'AdamW',
            'optimizer_args': {'lr': 3e-4, 'momentum': 0.9, 'weight_decay': 1e-5},
            'scheduler': 'StepLR',
            'scheduler_args': {'step_size': 4000, 'gamma': 0.1},
            'num_iters': 10000,
            'val_freq': 1000,
            
            'fp16': False # 18GB vs 23GB
        }
    )