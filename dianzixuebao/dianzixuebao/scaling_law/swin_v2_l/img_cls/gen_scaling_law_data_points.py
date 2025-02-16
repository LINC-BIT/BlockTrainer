from utils.dl.common.env import set_random_seed
import sys
# set_random_seed(5)
# set_random_seed(10) # 2nd run
set_random_seed(20) # 2nd run
# set_random_seed(int(sys.argv[1])) # 2nd run



import torch
import numpy as np
from utils.common.exp import get_res_save_dir, save_models_dict_for_init


if __name__ == '__main__':
    # 1. Data
    from dianzixuebao.settings.seg import semantic_segmentation_scenario as scenario
    
    # 2. Model(s)
    models_dict_path = save_models_dict_for_init(
        torch.load('dianzixuebao/offline_preparing/swinv2_l/seg/results/gen_neuron_index.py/20240510/999983-214633-after_fix_swinv2_window_bug/models/0.90/main_best_acc=0.3471.pt'),
        __file__
    )
    from dianzixuebao.model_impl.swin_v2_l.main import GenScalingLawDataPointsModel_SwinV2SemanticSegmentation
    model = GenScalingLawDataPointsModel_SwinV2SemanticSegmentation(
        name='swin_v2_l',
        models_dict_path=models_dict_path,
        device='cuda'
    )
    model.num_classes = scenario.num_classes
    
    # 3. Algorithm
    from methods.gen_scaling_law_data_points import GenScalingLawDataPointsAlg
    alg = GenScalingLawDataPointsAlg(
        models={
            'main': model
        },
        res_save_dir=get_res_save_dir(__file__, sys.argv[1])
    )
    
    qkv_layers = []
    proj_layers = []
    ff1_layers = []
    ff2_layers = []
    for si, stage in enumerate(model.model.swinv2.encoder.layers):
        for bi, block in enumerate(stage.blocks):
            qkv_layers += [[f'swinv2.encoder.layers.{si}.blocks.{bi}.attention.self.query', 
                           f'swinv2.encoder.layers.{si}.blocks.{bi}.attention.self.key', 
                           f'swinv2.encoder.layers.{si}.blocks.{bi}.attention.self.value']]
            proj_layers += [f'swinv2.encoder.layers.{si}.blocks.{bi}.attention.output.dense']
            ff1_layers += [f'swinv2.encoder.layers.{si}.blocks.{bi}.intermediate.dense']
            ff2_layers += [f'swinv2.encoder.layers.{si}.blocks.{bi}.output.dense']
    
    # 4. Run!
    from methods.feature_alignment import FeatureAlignmentAlg
    from dianzixuebao.model_impl.swin_v2_l.main import FeatureAlignmentModel_SwinV2SemanticSegmentation
    alg.run(
        scenario=scenario, 
        hyps={
            'sparsity': 0.875,
            'optional_batch_sizes': [16],
            'max_num_trials': 5000,
            
            'obtain_features_num_iters': 200,
            'obtain_larget_model_target_loss_num_iters': 1,
            
            'qkv_layers_name': qkv_layers,
            'proj_layers_name': proj_layers,
            'ff1_layers_name': ff1_layers,
            'ff2_layers_name': ff2_layers,
            
            'retraining_alg_cls': FeatureAlignmentAlg,
            'retraining_model_cls': FeatureAlignmentModel_SwinV2SemanticSegmentation,
            'retraining_hyps': {
                'train_batch_size': 16, # std batch_size
                'val_batch_size': 64, # num samples for evaluating retraining accuracy
                'num_workers': 16,
                'optimizer': 'AdamW',
                'optimizer_args': {'lr': 6e-5}, # std lr for batch_size=64
                'scheduler': '',
                'scheduler_args': {},
                'num_iters': 500,
                'val_freq': 10,
                'feat_align_loss_weight': 1.,
                'freeze_bn': False,
                'random_sim_policy': None,
                'auged_source_dataset_name': None,
                'fp16': False,
                'collate_fn': None
            }
        }
    )
    
    
    # alg.run_real_target(
    #     scenario=scenario, 
    #     hyps={
    #         'sparsity': 0.875,
    #         'optional_batch_sizes': [64],
    #         'max_num_trials': 5000,
            
    #         'obtain_features_num_iters': 200,
    #         'obtain_larget_model_target_loss_num_iters': 1,
            
    #         'qkv_layers_name': qkv_layers,
    #         'proj_layers_name': proj_layers,
    #         'ff1_layers_name': ff1_layers,
    #         'ff2_layers_name': ff2_layers,
            
    #         'retraining_alg_cls': FeatureAlignmentAlg,
    #         'retraining_model_cls': FeatureAlignmentModel_ViTImageClassification,
    #         'retraining_hyps': {
    #             'train_batch_size': 64, # std batch_size
    #             'val_batch_size': 1024, # num samples for evaluating retraining accuracy
    #             'num_workers': 16,
    #             'optimizer': 'AdamW',
    #             'optimizer_args': {'lr': 3e-4}, # std lr for batch_size=64
    #             'scheduler': '',
    #             'scheduler_args': {},
    #             'num_iters': 500,
    #             'val_freq': 10,
    #             'feat_align_loss_weight': 3.,
    #             'freeze_bn': False,
    #             'random_sim_policy': None,
    #             'auged_source_dataset_name': None,
    #             'fp16': True
    #         }
    #     }
    # )
    