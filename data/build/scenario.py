import enum
from functools import reduce
from typing import Dict, List, Tuple
import numpy as np
import copy
from utils.common.log import logger
from ..datasets.ab_dataset import ABDataset
from ..dataloader import FastDataLoader, InfiniteDataLoader, build_dataloader
from data import get_dataset


class DatasetMetaInfo:
    def __init__(self, name, 
                 known_classes_name_idx_map, unknown_class_idx):
        
        assert unknown_class_idx not in known_classes_name_idx_map.keys()
        
        self.name = name
        self.unknown_class_idx = unknown_class_idx
        self.known_classes_name_idx_map = known_classes_name_idx_map
        
    @property
    def num_classes(self):
        return len(self.known_classes_idx) + 1
        
        
class MergedDataset:
    def __init__(self, datasets: List[ABDataset]):
        self.datasets = datasets
        self.datasets_len = [len(i) for i in self.datasets]
        logger.info(f'create MergedDataset: len of datasets {self.datasets_len}')
        self.datasets_cum_len = np.cumsum(self.datasets_len)

    def __getitem__(self, idx):
        for i, cum_len in enumerate(self.datasets_cum_len):
            if idx < cum_len:
                return self.datasets[i][idx - sum(self.datasets_len[0: i])]
            
    def __len__(self):
        return sum(self.datasets_len)
    
    
class IndexReturnedDataset:
    def __init__(self, dataset: ABDataset):
        self.dataset = dataset
        
    def __getitem__(self, idx):
        res = self.dataset[idx]

        if isinstance(res, (tuple, list)):
            return (*res, idx)
        else:
            return res, idx
            
    def __len__(self):
        return len(self.dataset)
    

# class Scenario:
#     def __init__(self, config,
#                  source_datasets_meta_info: Dict[str, DatasetMetaInfo], target_datasets_meta_info: Dict[str, DatasetMetaInfo], 
#                  target_source_map: Dict[str, Dict[str, str]], 
#                  target_domains_order: List[str],
#                  source_datasets: Dict[str, Dict[str, ABDataset]], target_datasets: Dict[str, Dict[str, ABDataset]]):
        
#         self.__config = config
#         self.__source_datasets_meta_info = source_datasets_meta_info
#         self.__target_datasets_meta_info = target_datasets_meta_info
#         self.__target_source_map = target_source_map
#         self.__target_domains_order = target_domains_order
#         self.__source_datasets = source_datasets
#         self.__target_datasets = target_datasets
    
#     # 1. basic
#     def get_config(self):
#         return copy.deepcopy(self.__config)
    
#     def get_task_type(self):
#         return list(self.__source_datasets.values())[0]['train'].task_type
    
#     def get_num_classes(self):
#         known_classes_idx = []
#         unknown_classes_idx = []
#         for v in self.__source_datasets_meta_info.values():
#             known_classes_idx += list(v.known_classes_name_idx_map.values())
#             unknown_classes_idx += [v.unknown_class_idx]
#         for v in self.__target_datasets_meta_info.values():
#             known_classes_idx += list(v.known_classes_name_idx_map.values())
#             unknown_classes_idx += [v.unknown_class_idx]
#         unknown_classes_idx = [i for i in unknown_classes_idx if i is not None]
#         # print(known_classes_idx, unknown_classes_idx)
#         res = len(set(known_classes_idx)), len(set(unknown_classes_idx)), len(set(known_classes_idx + unknown_classes_idx))
#         # print(res)
#         assert res[0] + res[1] == res[2]
#         return res
      
#     def build_dataloader(self, dataset: ABDataset, batch_size: int, num_workers: int, infinite: bool, shuffle_when_finite: bool):
#         if infinite:
#             dataloader = InfiniteDataLoader(
#                 dataset, None, batch_size, num_workers=num_workers)
#         else:
#             dataloader = FastDataLoader(
#                 dataset, batch_size, num_workers, shuffle=shuffle_when_finite)

#         return dataloader
    
#     def build_sub_dataset(self, dataset: ABDataset, indexes: List[int]):
#         from ..data.datasets.dataset_split import _SplitDataset
#         dataset.dataset = _SplitDataset(dataset.dataset, indexes)
#         return dataset
    
#     def build_index_returned_dataset(self, dataset: ABDataset):
#         return IndexReturnedDataset(dataset)
        
#     # 2. source
#     def get_source_datasets_meta_info(self):
#         return self.__source_datasets_meta_info
    
#     def get_source_datasets_name(self):
#         return list(self.__source_datasets.keys())
    
#     def get_merged_source_dataset(self, split):
#         source_train_datasets = {n: d[split] for n, d in self.__source_datasets.items()}
#         return MergedDataset(list(source_train_datasets.values()))
    
#     def get_source_datasets(self, split):
#         source_train_datasets = {n: d[split] for n, d in self.__source_datasets.items()}
#         return source_train_datasets
    
#     # 3. target **domain**
#     # (do we need such API `get_ith_target_domain()`?)
#     def get_target_domains_meta_info(self):
#         return self.__source_datasets_meta_info
    
#     def get_target_domains_order(self):
#         return self.__target_domains_order
    
#     def get_corr_source_datasets_name_of_target_domain(self, target_domain_name):
#         return self.__target_source_map[target_domain_name]
    
#     def get_limited_target_train_dataset(self):
#         if len(self.__target_domains_order) > 1:
#             raise RuntimeError('this API is only for pass-in scenario in user-defined online DA algorithm')
#         return list(self.__target_datasets.values())[0]['train']
    
#     def get_target_domains_iterator(self, split):
#         for target_domain_index, target_domain_name in enumerate(self.__target_domains_order):
#             target_dataset = self.__target_datasets[target_domain_name]
#             target_domain_meta_info = self.__target_datasets_meta_info[target_domain_name]
            
#             yield target_domain_index, target_domain_name, target_dataset[split], target_domain_meta_info
    
#     # 4. permission management
#     def get_sub_scenario(self, source_datasets_name, source_splits, target_domains_order, target_splits):
#         def get_split(dataset, splits):
#             res = {}
#             for s, d in dataset.items():
#                 if s in splits:
#                     res[s] = d
#             return res
        
#         return Scenario(
#             config=self.__config,
#             source_datasets_meta_info={k: v for k, v in self.__source_datasets_meta_info.items() if k in source_datasets_name},
#             target_datasets_meta_info={k: v for k, v in self.__target_datasets_meta_info.items() if k in target_domains_order},
#             target_source_map={k: v for k, v in self.__target_source_map.items() if k in target_domains_order},
#             target_domains_order=target_domains_order,
#             source_datasets={k: get_split(v, source_splits) for k, v in self.__source_datasets.items() if k in source_datasets_name},
#             target_datasets={k: get_split(v, target_splits) for k, v in self.__target_datasets.items() if k in target_domains_order}
#         )
    
#     def get_only_source_sub_scenario_for_exp_tracker(self):
#         return self.get_sub_scenario(self.get_source_datasets_name(), ['train', 'val', 'test'], [], [])
    
#     def get_only_source_sub_scenario_for_alg(self):
#         return self.get_sub_scenario(self.get_source_datasets_name(), ['train'], [], [])
    
#     def get_one_da_sub_scenario_for_alg(self, target_domain_name):
#         return self.get_sub_scenario(self.get_corr_source_datasets_name_of_target_domain(target_domain_name), 
#                                      ['train', 'val'], [target_domain_name], ['train'])


# class Scenario:
#     def __init__(self, config,
                 
#                  offline_source_datasets_meta_info: Dict[str, DatasetMetaInfo], 
#                  offline_source_datasets: Dict[str, ABDataset],
                 
#                  online_datasets_meta_info: List[Tuple[Dict[str, DatasetMetaInfo], DatasetMetaInfo]],
#                  online_datasets: Dict[str, ABDataset],
#                  target_domains_order: List[str],
#                  target_source_map: Dict[str, Dict[str, str]],
                 
#                  num_classes: int):
        
#         self.config = config
        
#         self.offline_source_datasets_meta_info = offline_source_datasets_meta_info
#         self.offline_source_datasets = offline_source_datasets
        
#         self.online_datasets_meta_info = online_datasets_meta_info
#         self.online_datasets = online_datasets
        
#         self.target_domains_order = target_domains_order
#         self.target_source_map = target_source_map
        
#         self.num_classes = num_classes
        
#     def get_offline_source_datasets(self, split):
#         return {n: d[split] for n, d in self.offline_source_datasets.items()}
    
#     def get_offline_source_merged_dataset(self, split):
#         return MergedDataset([d[split] for d in self.offline_source_datasets.values()])
    
#     def get_online_current_corresponding_source_datasets(self, domain_index, split):
#         cur_target_domain_name = self.target_domains_order[domain_index]
#         cur_source_datasets_name = list(self.target_source_map[cur_target_domain_name].keys())
#         cur_source_datasets = {n: self.online_datasets[n + '|' + cur_target_domain_name][split] for n in cur_source_datasets_name}
#         return cur_source_datasets
    
#     def get_online_current_corresponding_merged_source_dataset(self, domain_index, split):
#         cur_target_domain_name = self.target_domains_order[domain_index]
#         cur_source_datasets_name = list(self.target_source_map[cur_target_domain_name].keys())
#         cur_source_datasets = {n: self.online_datasets[n + '|' + cur_target_domain_name][split] for n in cur_source_datasets_name}
#         return MergedDataset([d for d in cur_source_datasets.values()])
    
#     def get_online_current_target_dataset(self, domain_index, split):
#         cur_target_domain_name = self.target_domains_order[domain_index]
#         return self.online_datasets[cur_target_domain_name][split]
    
#     def build_dataloader(self, dataset: ABDataset, batch_size: int, num_workers: int, 
#                          infinite: bool, shuffle_when_finite: bool, to_iterator: bool):
#         if infinite:
#             dataloader = InfiniteDataLoader(
#                 dataset, None, batch_size, num_workers=num_workers)
#         else:
#             dataloader = FastDataLoader(
#                 dataset, batch_size, num_workers, shuffle=shuffle_when_finite)
            
#         if to_iterator:
#             dataloader = iter(dataloader)

#         return dataloader
    
#     def build_sub_dataset(self, dataset: ABDataset, indexes: List[int]):
#         from data.datasets.dataset_split import _SplitDataset
#         dataset.dataset = _SplitDataset(dataset.dataset, indexes)
#         return dataset
    
#     def build_index_returned_dataset(self, dataset: ABDataset):
#         return IndexReturnedDataset(dataset)
    
#     def get_config(self):
#         return copy.deepcopy(self.config)
    
#     def get_task_type(self):
#         return list(self.online_datasets.values())[0]['train'].task_type
    
#     def get_num_classes(self):
#         return self.num_classes
    

class Scenario:
    def __init__(self, config, all_datasets_ignore_classes_map, all_datasets_idx_map, target_domains_order, target_source_map, 
                 all_datasets_e2e_class_to_idx_map,
                 num_classes):
        self.config = config 
        self.all_datasets_ignore_classes_map = all_datasets_ignore_classes_map
        self.all_datasets_idx_map = all_datasets_idx_map
        self.target_domains_order = target_domains_order
        self.target_source_map = target_source_map
        self.all_datasets_e2e_class_to_idx_map = all_datasets_e2e_class_to_idx_map
        self.num_classes = num_classes
        self.cur_domain_index = 0

        logger.info(f'[scenario build] # classes: {num_classes}')
        logger.debug(f'[scenario build] idx map: {all_datasets_idx_map}')
        
    def to_json(self):
        return dict(
            config=self.config, all_datasets_ignore_classes_map=self.all_datasets_ignore_classes_map,
            all_datasets_idx_map=self.all_datasets_idx_map, target_domains_order=self.target_domains_order,
            target_source_map=self.target_source_map, 
            all_datasets_e2e_class_to_idx_map=self.all_datasets_e2e_class_to_idx_map,
            num_classes=self.num_classes
        )
        
    def __str__(self):
        return f'Scenario({self.to_json()})'

    def get_offline_datasets(self, use_before_res=False):
        
        if use_before_res and hasattr(self, 'res_offline_train_source_datasets_map'):
            logger.info('use before constructed offline datasets')
            return self.res_offline_train_source_datasets_map
        
        # make source datasets which contains all unioned classes
        res_offline_train_source_datasets_map = {}

        from .. import get_dataset
        data_dirs = self.config['data_dirs']
        transforms = self.config['transforms']

        source_datasets_name = self.config['source_datasets_name']
        res_source_datasets_map = {d: {split: get_dataset(d.split('|')[0], data_dirs[d.split('|')[0]], split, 
                                                      transforms[d.split('|')[0]],
                                                      self.all_datasets_ignore_classes_map[d], self.all_datasets_idx_map[d]) 
                                   for split in ['train', 'val', 'test']} 
                               for d in self.all_datasets_ignore_classes_map.keys() if d.split('|')[0] in source_datasets_name}
        
        for source_dataset_name in self.config['source_datasets_name']:
            source_datasets = [v for k, v in res_source_datasets_map.items() if source_dataset_name in k]

            # how to merge idx map?
            # 35 79 97
            idx_maps = [d['train'].idx_map for d in source_datasets]
            ignore_classes_list = [d['train'].ignore_classes for d in source_datasets]
            
            union_idx_map = {}
            for idx_map in idx_maps:
                for k, v in idx_map.items():
                    if k not in union_idx_map:
                        union_idx_map[k] = v
                    else:
                        assert union_idx_map[k] == v

            union_ignore_classes = reduce(lambda res, cur: res & set(cur), ignore_classes_list, set(ignore_classes_list[0]))
            assert len(union_ignore_classes) + len(union_idx_map) == len(source_datasets[0]['train'].raw_classes)

            logger.info(f'[scenario build] {source_dataset_name} has {len(union_idx_map)} classes in offline training')
            
            d = source_dataset_name
            res_offline_train_source_datasets_map[d] = {split: get_dataset(d, data_dirs[d], split, 
                                                      transforms[d],
                                                      union_ignore_classes, union_idx_map) 
                                   for split in ['train', 'val', 'test']} 
        
        self.res_offline_train_source_datasets_map = res_offline_train_source_datasets_map
        
        return res_offline_train_source_datasets_map
    
    def get_offline_datasets_args(self):
        # make source datasets which contains all unioned classes
        res_offline_train_source_datasets_map = {}

        from .. import get_dataset
        data_dirs = self.config['data_dirs']
        transforms = self.config['transforms']

        source_datasets_name = self.config['source_datasets_name']
        res_source_datasets_map = {d: {split: get_dataset(d.split('|')[0], data_dirs[d.split('|')[0]], split, 
                                                      transforms[d.split('|')[0]],
                                                      self.all_datasets_ignore_classes_map[d], self.all_datasets_idx_map[d]) 
                                   for split in ['train', 'val', 'test']} 
                               for d in self.all_datasets_ignore_classes_map.keys() if d.split('|')[0] in source_datasets_name}
        
        for source_dataset_name in self.config['source_datasets_name']:
            source_datasets = [v for k, v in res_source_datasets_map.items() if source_dataset_name in k]

            # how to merge idx map?
            # 35 79 97
            idx_maps = [d['train'].idx_map for d in source_datasets]
            ignore_classes_list = [d['train'].ignore_classes for d in source_datasets]
            
            union_idx_map = {}
            for idx_map in idx_maps:
                for k, v in idx_map.items():
                    if k not in union_idx_map:
                        union_idx_map[k] = v
                    else:
                        assert union_idx_map[k] == v

            union_ignore_classes = reduce(lambda res, cur: res & set(cur), ignore_classes_list, set(ignore_classes_list[0]))
            assert len(union_ignore_classes) + len(union_idx_map) == len(source_datasets[0]['train'].raw_classes)

            logger.info(f'[scenario build] {source_dataset_name} has {len(union_idx_map)} classes in offline training')
            
            d = source_dataset_name
            res_offline_train_source_datasets_map[d] = {split: dict(d, data_dirs[d], split, 
                                                      transforms[d],
                                                      union_ignore_classes, union_idx_map) 
                                   for split in ['train', 'val', 'test']} 

        return res_offline_train_source_datasets_map

        # for d in source_datasets_name:
        #     source_dataset_with_max_num_classes = None
            
        #     for ed_name, ed in res_source_datasets_map.items():
        #         if not ed_name.startswith(d):
        #             continue
                
        #         if source_dataset_with_max_num_classes is None:
        #             source_dataset_with_max_num_classes = ed
        #             res_offline_train_source_datasets_map_names[d] = ed_name
                    
        #         if len(ed['train'].ignore_classes) < len(source_dataset_with_max_num_classes['train'].ignore_classes):
        #             source_dataset_with_max_num_classes = ed
        #             res_offline_train_source_datasets_map_names[d] = ed_name
                    
        #     res_offline_train_source_datasets_map[d] = source_dataset_with_max_num_classes

        # return res_offline_train_source_datasets_map
        
    def get_online_ith_domain_datasets_args_for_inference(self, domain_index):
        target_dataset_name = self.target_domains_order[domain_index]
        # dataset_name: Any, root_dir: Any, split: Any, transform: Any | None = None, ignore_classes: Any = [], idx_map: Any | None = None
        
        if 'MM-CityscapesDet' in self.target_domains_order or 'CityscapesDet' in self.target_domains_order or 'BaiduPersonDet' in self.target_domains_order:
            logger.info(f'use val split for inference test (only Det workload)')
            split = 'test'
        else:
            split = 'train'
            
        transforms = self.config['transforms']
        
        return dict(dataset_name=target_dataset_name, 
                    root_dir=self.config['data_dirs'][target_dataset_name], 
                    split=split, 
                    transform=transforms[target_dataset_name], 
                    ignore_classes=self.all_datasets_ignore_classes_map[target_dataset_name], 
                    idx_map=self.all_datasets_idx_map[target_dataset_name])
    
    def get_online_ith_domain_datasets_args_for_training(self, domain_index):
        target_dataset_name = self.target_domains_order[domain_index]
        source_datasets_name = list(self.target_source_map[target_dataset_name].keys())
        
        transforms = self.config['transforms']

        res = {}
        # dataset_name: Any, root_dir: Any, split: Any, transform: Any | None = None, ignore_classes: Any = [], idx_map: Any | None = None
        res[target_dataset_name] = {split: dict(dataset_name=target_dataset_name, 
                    root_dir=self.config['data_dirs'][target_dataset_name], 
                    split=split, 
                    transform=transforms[target_dataset_name], 
                    ignore_classes=self.all_datasets_ignore_classes_map[target_dataset_name], 
                    idx_map=self.all_datasets_idx_map[target_dataset_name]) for split in ['train', 'val']}
        for d in source_datasets_name:
            res[d] = {split: dict(dataset_name=d, 
                    root_dir=self.config['data_dirs'][d], 
                    split=split, 
                    transform=transforms[target_dataset_name], 
                    ignore_classes=self.all_datasets_ignore_classes_map[d + '|' + target_dataset_name], 
                    idx_map=self.all_datasets_idx_map[d + '|' + target_dataset_name]) for split in ['train', 'val']}
        
        return res
    
    def get_online_cur_domain_datasets_args_for_inference(self):
        return self.get_online_ith_domain_datasets_args_for_inference(self.cur_domain_index)
    
    def get_online_cur_domain_datasets_args_for_training(self):
        return self.get_online_ith_domain_datasets_args_for_training(self.cur_domain_index)
    
    def get_online_cur_domain_datasets_for_training(self):
        res = {}
        datasets_args = self.get_online_ith_domain_datasets_args_for_training(self.cur_domain_index)
        for dataset_name, dataset_args in datasets_args.items():
            res[dataset_name] = {}
            for split, args in dataset_args.items():
                dataset = get_dataset(**args)
                res[dataset_name][split] = dataset
        return res
    
    def get_online_cur_domain_datasets_for_inference(self):
        datasets_args = self.get_online_ith_domain_datasets_args_for_inference(self.cur_domain_index)
        return get_dataset(**datasets_args)
    
    def get_online_cur_domain_samples_for_training(self, num_samples, collate_fn=None):
        dataset = self.get_online_cur_domain_datasets_for_training()
        dataset = dataset[self.target_domains_order[self.cur_domain_index]]['train']
        return next(iter(build_dataloader(dataset, num_samples, 0, True, None, collate_fn=collate_fn)))[0]

    def next_domain(self):
        self.cur_domain_index += 1
        