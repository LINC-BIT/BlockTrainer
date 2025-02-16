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
import numpy as np

from utils.common.others import HiddenPrints
from utils.dl.common.model import get_module, set_module, get_model_size, LayerActivation2, LayerActivationWithCustomFunc
from utils.common.log import logger

from torch.cuda.amp import autocast

import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import json



class PretrainOrFineTuningModel_TextGeneration(PretrainOrFineTuningModel):
    def get_required_model_components(self) -> List[str]:
        return ['main']
    
    def get_accuracy(self, test_loader, *args, **kwargs):
        acc = 0
        sample_num = 0
        # tokenizer = getTokenizer()
        
        if hasattr(self, 'tokenizer'):
            tokenizer = self.tokenizer
        else:
            from dnns.llama import AttentionPrunableLlamaForCausalLM, get_tokenizer
            tokenizer = get_tokenizer()
            self.tokenizer = tokenizer
            
        torch.cuda.empty_cache()
        
        self.to_eval_mode()
        pred_txt = []
        true_txt = []
        res = []
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, _) in pbar:
                if len(x) == 0:
                    continue
                # if batch_index > 10:
                #     break
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(self.device)
                # input_ids = []
                # inputlen = x['len']
                y = x['labels']
                x['labels'] = None
                outputs = self.models_dict['main'].generate(x['input_ids'], max_new_tokens=128, num_beams=1, 
                                                            early_stopping=True, 
                                                            pad_token_id=tokenizer.pad_token_id)
                
                len_inp = len(list(filter(lambda x: x != tokenizer.pad_token_id, x['input_ids'].squeeze(0).tolist())))
                
                for i, op in enumerate(outputs):
                    op = op.tolist()
                    op = list(filter(lambda x: x != tokenizer.pad_token_id, op))
                    txt = tokenizer.decode(op)
                    txt = txt.replace(tokenizer.bos_token + " ", "")
                    res.append(txt)
                    txt = tokenizer.decode(op[len_inp:])
                    pred_txt.append(nltk.word_tokenize(txt))
                

                for tp in y:
                    # _tp = deepcopy(tp)
                    # tp = tp.numpy()
                    # https://blog.csdn.net/qq_43576728/article/details/131918828
                    tp = torch.where(tp != -100, tp, self.tokenizer.pad_token_id)
                    # tp = 
                    true_txt.append(nltk.word_tokenize(tokenizer.decode(tp).replace(tokenizer.pad_token, '').replace(tokenizer.bos_token, '')))
                # pred = F.softmax(output, dim=1).argmax(dim=1)
                # correct = torch.eq(pred, y).sum().item()
                # acc += correct
                sample_num += len(y)
                
                # pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_correct: {correct}, '
                #                      f'cur_batch_acc: {(correct / len(y)):.4f}')
        json.dump(res, open("./llama_generation.json", "w"))
        smooth = SmoothingFunction()
        score = 0.
        for pred, true in zip(pred_txt, true_txt):
            score += sentence_bleu([true], pred, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
        score /= sample_num
        
        torch.cuda.empty_cache()
        
        return score
        
    def infer(self, x, *args, **kwargs):
        # if hasattr(self, 'tokenizer'):
        #     tokenizer = self.tokenizer
        # else:
        #     from dnns.llama import AttentionPrunableLlamaForCausalLM, get_tokenizer
        #     tokenizer = get_tokenizer()
        #     self.tokenizer = tokenizer
        x['return_dict'] = True
        x['labels'] = None
        return self.model(**x).logits
        # return self.models_dict['main'].generate(x['input_ids'], max_new_tokens=128, num_beams=1, 
        #                                                     early_stopping=True, 
        #                                                     pad_token_id=tokenizer.pad_token_id)
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        x['return_dict'] = True
        return self.model(**x).loss
    
    
class TrainWithFBSModel_TextGeneration(TrainWithFBSModel, PretrainOrFineTuningModel_TextGeneration):
    pass


class LoRAFineTuningModel_TextGeneration(LoRAFineTuningModel, PretrainOrFineTuningModel_TextGeneration):
    def get_head_params(self):
        return self.model.lm_head.parameters()

class GenKnowledgeBaseModel_TextGeneration(GenKnowledgeBaseModel, PretrainOrFineTuningModel_TextGeneration):
    pass

class GenNeuronIndexModel_TextGeneration(GenNeuronIndexModel, PretrainOrFineTuningModel_TextGeneration):
    pass


class GenScalingLawDataPointsModel_TextGeneration(GenScalingLawDataPointsModel, PretrainOrFineTuningModel_TextGeneration):
    def get_output_entropy(self, samples):
        # x = self.infer(samples)
        # return -(x.softmax(1) * x.log_softmax(1)).sum(1)
        return torch.rand(len(samples), device=self.device).float()
    
    def guess_linear_name(self):
        cand_linear_names = ['fc', 'linear', 'classifier', 'lm_head']
        for name in cand_linear_names:
            if get_module(self.model, name) is not None:
                return name
        raise NotImplementedError
    
    def get_2d_feature(self, raw_feature):
        return raw_feature[0].mean(1)
    
    def get_feature_hook(self):
        return LayerActivationWithCustomFunc(get_module(self.model, self.guess_linear_name()), self.get_2d_feature, self.get_2d_feature)
    
    
class FeatureAlignmentModel_TextGeneration(FeatureAlignmentModel, PretrainOrFineTuningModel_TextGeneration):
    def guess_linear_name(self):
        cand_linear_names = ['fc', 'linear', 'classifier', 'lm_head']
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
    
    def get_params_names_of_each_block(self):
        pass
    
    
class FeatureAlignmentModel_TextGeneration_Profile(FeatureAlignmentModel, PretrainOrFineTuningModel_TextGeneration):
    def guess_linear_name(self):
        cand_linear_names = ['fc', 'linear', 'classifier', 'lm_head']
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
    
    def get_params_names_of_each_block(self):
        pass
    
    
class FeatureAlignmentLoraModel_TextGeneration(FeatureAlignmentLoraModel, FeatureAlignmentModel_TextGeneration):
    pass


class EdgeTAOnlineRunModel_TextGeneration(EdgeTAOnlineRunModel, GenScalingLawDataPointsModel_TextGeneration):
    pass