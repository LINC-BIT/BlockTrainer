from typing import List

from torch.nn.modules import Module
from methods.pretrain_or_ft import PretrainOrFineTuningModel
from methods.train_with_fbs import TrainWithFBSModel
from methods.lora_ft import LoRAFineTuningModel
from methods.gen_knowledge_base import GenKnowledgeBaseModel
from methods.gen_neuron_index import GenNeuronIndexModel
from methods.gen_scaling_law_data_points import GenScalingLawDataPointsModel
from methods.train_with_fbs.lib import clear_cache
from methods.simply_retrain_small_model import SimplyRetrainSmallModelModel
from methods.feature_alignment import FeatureAlignmentModel
from methods.feature_alignment_lora.model import FeatureAlignmentLoraModel
from methods.edgeta_online_run.model import EdgeTAOnlineRunModel
import torch
import tqdm
import copy
import random
from torch import nn
import torch.nn.functional as F

from utils.common.others import HiddenPrints
from utils.dl.common.model import get_module, set_module, get_model_size, LayerActivation2, LayerActivationWithCustomFunc
from utils.common.log import logger

from torch.cuda.amp import autocast


class PretrainOrFineTuningModel_OPTTokenClassification(PretrainOrFineTuningModel):
    def get_required_model_components(self) -> List[str]:
        return ['main']
    
    def get_accuracy(self, test_loader, *args, **kwargs):
        acc = 0
        sample_num = 0
        
        self.to_eval_mode()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y) in pbar:
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(self.device)
                # print(x)
                y = y.to(self.device)
                # output = self.infer(x)
                with autocast(self.fp16, dtype=torch.float16):
                    output = self.infer(x)
                
                # torch.Size([16, 512, 43]) torch.Size([16, 512])
                
                for oi, yi, xi in zip(output, y, x['input_ids']):
                    # oi: 512, 43; yi: 512
                    # seq_len = xi.nonzero().size(0)
                    # print(xi, torch.where(xi == 2))
                    seq_len = torch.where(yi == -100)[0]
                    if len(seq_len) == 0:
                        seq_len = len(xi)
                    else:
                        seq_len = seq_len[0]
                    
                    # print(output.size(), y.size())
                    # print(oi, yi, xi, oi.size(), yi.size(), xi.size(), seq_len)
                    # exit()
                    
                    pred = F.softmax(oi, dim=-1).argmax(dim=-1)
                    correct = torch.eq(pred[1: seq_len + 1], yi[0: seq_len]).sum().item()
                    
                    # print(output.size(), y.size())
                    
                    acc += correct
                    sample_num += seq_len
                
                    pbar.set_description(f'seq_len: {seq_len}, cur_seq_acc: {(correct / seq_len):.4f}')
        acc = float(acc)
        acc /= sample_num
        return acc
        
    def infer(self, x, *args, **kwargs):
        x['return_dict'] = True
        return self.model(**x).logits
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        logits = self.infer(x)
        labels = y
        
        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_labels = labels[..., 1:].contiguous()
        shift_logits = logits[..., 1:, :].contiguous()
        shift_labels = labels[..., :-1].contiguous()
        
        return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    
class TrainWithFBSModel_OPTTokenClassification(TrainWithFBSModel, PretrainOrFineTuningModel_OPTTokenClassification):
    pass


class LoRAFineTuningModel_OPTTokenClassification(LoRAFineTuningModel, PretrainOrFineTuningModel_OPTTokenClassification):
    def get_head_params(self):
        return self.model.classifier.parameters()

class GenKnowledgeBaseModel_OPTTokenClassification(GenKnowledgeBaseModel, PretrainOrFineTuningModel_OPTTokenClassification):
    pass

class GenNeuronIndexModel_OPTTokenClassification(GenNeuronIndexModel, PretrainOrFineTuningModel_OPTTokenClassification):
    pass


class GenScalingLawDataPointsModel_OPTTokenClassification(GenScalingLawDataPointsModel, PretrainOrFineTuningModel_OPTTokenClassification):
    def get_output_entropy(self, samples):
        # x = self.infer(samples)
        # return -(x.softmax(1) * x.log_softmax(1)).sum(1)
        return torch.rand(len(samples), device=self.device).float()
    
    def guess_linear_name(self):
        cand_linear_names = ['fc', 'linear', 'classifier']
        for name in cand_linear_names:
            if get_module(self.model, name) is not None:
                return name
        raise NotImplementedError
    
    def get_2d_feature(self, raw_feature):
        return raw_feature[0].mean(1)
    
    def get_feature_hook(self):
        return LayerActivationWithCustomFunc(get_module(self.model, self.guess_linear_name()), self.get_2d_feature, self.get_2d_feature)
    
    
class FeatureAlignmentModel_OPTTokenClassification(FeatureAlignmentModel, PretrainOrFineTuningModel_OPTTokenClassification):
    def guess_linear_name(self):
        cand_linear_names = ['fc', 'linear', 'classifier']
        for name in cand_linear_names:
            if get_module(self.model, name) is not None:
                return name
        raise NotImplementedError
    
    def get_2d_feature(self, raw_feature):
        # print(raw_feature, raw_feature.size())
        # exit()
        return raw_feature[0].mean(1)
    
    def get_feature_hook(self):
        return LayerActivationWithCustomFunc(get_module(self.model, self.guess_linear_name()), self.get_2d_feature, self.get_2d_feature)
    
    def get_trained_params(self):
        res = []
        
        linear_name = self.guess_linear_name()
        for name, param in self.model.named_parameters():
            if name.startswith(linear_name):
                continue
            res += [param]
        return res
    
    
class FeatureAlignmentLoraModel_OPTTokenClassification(FeatureAlignmentLoraModel, FeatureAlignmentModel_OPTTokenClassification):
    pass


class EdgeTAOnlineRunModel_OPTTokenClassification(EdgeTAOnlineRunModel, GenScalingLawDataPointsModel_OPTTokenClassification):
    pass