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
    fm = torch.load('dianzixuebao/offline_preparing/opt_1_3b/results/lora_fine_tune.py/20240515/999998-200637-new_loss/models/main_best_acc=0.7795.pt')
    from dnns.llama import AttentionPrunableLlamaForCausalLM, get_tokenizer
    tokenizer = get_tokenizer()
    # model = AttentionPrunableLlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b_v2")
    # model.config.pad_token_id = model.config.eos_token_id
    # model.resize_token_embeddings(len(tokenizer))
    # model.tie_weights()
    # fm = model
    # fm = nn.Linear(10, 10)
    
    knowledge_base = torch.load('dianzixuebao/offline_preparing/llama/results/gen_neuron_index.py/20240519/999982-135953-trial/models/0.90/main_best_acc=0.0210.pt')
    neuron_index = torch.load('dianzixuebao/offline_preparing/llama/results/gen_neuron_index.py/20240519/999982-135953-trial/models/0.90/best_neuron_index.pt')
    from analysis.scaling_law_trial.only_consider_pruned_block_index2 import *
    # scaling_law = torch.load('analysis/scaling_law_trial/results/only_consider_pruned_block_index_for_swin.py/20240514/999999-093418-trial/best_edge_scaling_law_fcn.pt')
    scaling_law = None
    
    
    models_dict = {
        'fm': fm,
        'knowledge_base': knowledge_base['main'],
        'bn_stats': {},
        'neuron_index': neuron_index,
        'scaling_law': scaling_law,
        'profiles': {
            'blocks_retraining_time_per_iteration': [[2.779/24, 2.666/24]] * 24,
            'num_blocks_params': [[p1, p2] for p1, p2 in zip([68897, 68897, 205313, 205313, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 3535494, 3535494], [i * 0.125 for i in [68897, 68897, 205313, 205313, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 902211, 3535494, 3535494]])],
        }
    }
    
    # models_dict_path = save_models_dict_for_init(
    #     models_dict,
    #     __file__, uid='vit_b_16_img_cls_edgeta_da_model'
    # )
    
    from dianzixuebao.model_impl.llama.main import EdgeTAOnlineRunModel_TextGeneration, FeatureAlignmentModel_TextGeneration
    da_model = EdgeTAOnlineRunModel_TextGeneration(
        name='vit_b_16_img_cls_edgeta_da_model',
        models_dict_path=models_dict,
        device='cuda'
    )
    
    # 3. Algorithm
    from methods.edgeta_online_run import EdgeTAOnlineRunAlg
    from methods.feature_alignment.alg import FeatureAlignmentAlg
    
    # 4. Run!
    qkv_layers = [[f'model.layers.{i}.self_attn.q_proj', 
                   f'model.layers.{i}.self_attn.k_proj', 
                   f'model.layers.{i}.self_attn.v_proj'] 
                  for i in range(26)]
    proj_layers = [f'model.layers.{i}.self_attn.o_proj' for i in range(24)]
    ff1_layers = [[f'model.layers.{i}.mlp.gate_proj', f'model.layers.{i}.mlp.up_proj'] for i in range(26)]
    ff2_layers = [f'model.layers.{i}.mlp.down_proj' for i in range(26)]
    
    from dianzixuebao.online_retraining.da import da_exp
    
    da_exp(
        app_name='vit-b-16-img-cls-da-edgeta',
        scenario=scenario,
        da_alg=EdgeTAOnlineRunAlg,
        da_alg_hyp={
            'sparsity': 0.6,
            'obtain_features_num_iters': 1,
            'obtain_larget_model_target_loss_num_iters': 1,
            
            'qkv_layers_name': qkv_layers,
            'proj_layers_name': proj_layers,
            'ff1_layers_name': ff1_layers,
            'ff2_layers_name': ff2_layers,
            
            'retraining_alg_cls': FeatureAlignmentAlg,
            'retraining_model_cls': FeatureAlignmentModel_TextGeneration,
            'retraining_hyps': {
                'train_batch_size': 3, # std batch_size
                'val_batch_size': 16, # num samples for evaluating retraining accuracy
                'num_workers': 16,
                'optimizer': 'AdamW',
                'optimizer_args': {'lr': 1e-4}, # std lr for batch_size=64
                'scheduler': '',
                'scheduler_args': {},
                'num_iters': 500,
                'val_freq': 10,
                'feat_align_loss_weight': 1.,
                'freeze_bn': False,
                'random_sim_policy': None,
                'auged_source_dataset_name': None,
                'fp16': True,
                'collate_fn': collate_fn
            },
            'retraining_window': 1200,
            'knowledge_base_to_fm_alpha': 0.1,
            'fm_to_knowledge_base_alpha': 0.1,
            # 'opt_strategy': f'heuristic',
            'opt_strategy': ([True] * 26, 500),
            'up_bound_pred_acc': 0.7
        },
        da_model=da_model,
        device='cuda',
        __entry_file__=__file__,
        tag=f'heuristic',
        use_entry_model_in_new_dist=False
    )
    