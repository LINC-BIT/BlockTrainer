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
from utils.dl.common.model import LayerActivationWithCustomFunc, get_module, set_module, get_model_size, LayerActivation2
from utils.common.log import logger

from torch.cuda.amp import autocast


import numpy as np
class StreamSegMetrics:
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))



class PretrainOrFineTuningModel_SwinV2SemanticSegmentation(PretrainOrFineTuningModel):
    def get_required_model_components(self) -> List[str]:
        return ['main']
    
    def get_accuracy(self, test_loader, *args, **kwargs):
        device = self.device
        self.to_eval_mode()
        metrics = StreamSegMetrics(self.num_classes)
        metrics.reset()
        import tqdm
        pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for batch_index, (x, y) in pbar:
                x, y = x.to(device, dtype=x.dtype, non_blocking=True, copy=False), \
                    y.to(device, dtype=y.dtype, non_blocking=True, copy=False)
                    
                with autocast(self.fp16, dtype=torch.float16):
                    output = self.infer(x)
                # output = self.infer(x)
                pred = output.detach().max(dim=1)[1].cpu().numpy()
                metrics.update((y + 0).cpu().numpy(), pred)
                
                res = metrics.get_results()
                pbar.set_description(f'cur batch mIoU: {res["Mean IoU"]:.4f}')
                
        res = metrics.get_results()
        return res['Mean IoU']
        
    def infer(self, x, *args, **kwargs):
        return self.model(pixel_values=x).logits
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        return F.cross_entropy(self.infer(x), y)
    
    
class TrainWithFBSModel_SwinV2SemanticSegmentation(TrainWithFBSModel, PretrainOrFineTuningModel_SwinV2SemanticSegmentation):
    pass


class LoRAFineTuningModel_SwinV2SemanticSegmentation(LoRAFineTuningModel, PretrainOrFineTuningModel_SwinV2SemanticSegmentation):
    def get_head_params(self):
        return self.model.classifier.parameters()
    

class GenKnowledgeBaseModel_SwinV2SemanticSegmentation(GenKnowledgeBaseModel, PretrainOrFineTuningModel_SwinV2SemanticSegmentation):
    pass


class GenNeuronIndexModel_SwinV2SemanticSegmentation(GenNeuronIndexModel, PretrainOrFineTuningModel_SwinV2SemanticSegmentation):
    pass


class GenScalingLawDataPointsModel_SwinV2SemanticSegmentation(GenScalingLawDataPointsModel, PretrainOrFineTuningModel_SwinV2SemanticSegmentation):
    def get_output_entropy(self, samples):
        output = self.infer(samples)
        raw_size = output.size()
        
        output = output.permute(0, 2, 3, 1) # (B, H, W, C)
        # print(output.size())
        output = output.view(-1, output.size(3)) # (B*H*W, C)
        # print(output.size())
        
        def Entropy(input_):
            entropy = -input_ * torch.log(input_ + 1e-5)
            entropy = torch.sum(entropy, dim=1)
            return entropy 

        softmax_out = nn.Softmax(dim=1)(output)
        entropy_loss = Entropy(softmax_out).view(raw_size[0], -1)
        entropy_loss = entropy_loss.mean(1)
        return entropy_loss
    
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
    
    
class FeatureAlignmentModel_SwinV2SemanticSegmentation(FeatureAlignmentModel, PretrainOrFineTuningModel_SwinV2SemanticSegmentation):
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
    
    def get_trained_params(self):
        res = []
        
        linear_name = self.guess_linear_name()
        for name, param in self.model.named_parameters():
            if name.startswith(linear_name):
                continue
            res += [param]
        return res
    
    def get_params_names_of_prefix(self, prefix):
        return [n for n, _ in self.model.named_parameters() if n.startswith(prefix)]
    
    def get_params_names_of_each_block(self):
        
        blocks_prefix = []
        for si, stage in enumerate(self.model.swinv2.encoder.layers):
            for bi, block in enumerate(stage.blocks):
                blocks_prefix += [f'swinv2.encoder.layers.{si}.blocks.{bi}']
        
        return [self.get_params_names_of_prefix('swinv2.embeddings')] + \
            [self.get_params_names_of_prefix(p) for p in blocks_prefix] + \
            [self.get_params_names_of_prefix('classifier')]
    
    
class FeatureAlignmentLoraModel_SwinV2SemanticSegmentation(FeatureAlignmentLoraModel, FeatureAlignmentModel_SwinV2SemanticSegmentation):
    pass


class EdgeTAOnlineRunModel_SwinV2SemanticSegmentation(EdgeTAOnlineRunModel, GenScalingLawDataPointsModel_SwinV2SemanticSegmentation):
    pass