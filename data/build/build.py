from typing import Dict, List, Optional, Type, Union
from ..datasets.ab_dataset import ABDataset
# from benchmark.data.visualize import visualize_classes_in_object_detection
# from benchmark.scenario.val_domain_shift import get_val_domain_shift_transform
from ..dataset import get_dataset
import copy
from torchvision.transforms import Compose

from .merge_alias import merge_the_same_meaning_classes
from ..datasets.registery import static_dataset_registery


# some legacy aliases of variables:
# ignore_classes == discarded classes
# private_classes == unknown classes in partial / open-set / universal DA


def _merge_the_same_meaning_classes(classes_info_of_all_datasets):
    final_classes_of_all_datasets, rename_map = merge_the_same_meaning_classes(classes_info_of_all_datasets)
    return final_classes_of_all_datasets, rename_map


def _find_ignore_classes_when_sources_as_to_target_b(as_classes: List[List[str]], b_classes: List[str], da_mode):
    thres = {'da': 3, 'partial_da': 2, 'open_set_da': 1, 'universal_da': 0}[da_mode]
    
    from functools import reduce
    a_classes = reduce(lambda res, cur: res | set(cur), as_classes, set())
    
    if set(a_classes) == set(b_classes):
        # a is equal to b, normal
        # 1. no ignore classes; 2. match class idx
        a_ignore_classes, b_ignore_classes = [], []
    
    elif set(a_classes) > set(b_classes):
        # a contains b, partial
        a_ignore_classes, b_ignore_classes = [], []
        if thres == 3 or thres == 1: # ignore extra classes in a
            a_ignore_classes = set(a_classes) - set(b_classes)
        
    elif set(a_classes) < set(b_classes):
        # a is contained by b, open set
        a_ignore_classes, b_ignore_classes = [], []
        if thres == 3 or thres == 2: # ignore extra classes in b
            b_ignore_classes = set(b_classes) - set(a_classes)
    
    elif len(set(a_classes) & set(b_classes)) > 0:
        a_ignore_classes, b_ignore_classes = [], []
        if thres == 3:
            a_ignore_classes = set(a_classes) - (set(a_classes) & set(b_classes))
            b_ignore_classes = set(b_classes) - (set(a_classes) & set(b_classes))
        elif thres == 2:
            b_ignore_classes = set(b_classes) - (set(a_classes) & set(b_classes))
        elif thres == 1:
            a_ignore_classes = set(a_classes) - (set(a_classes) & set(b_classes))
    
    else:
        return None # a has no intersection with b, none
    
    as_ignore_classes = [list(set(a_classes) & set(a_ignore_classes)) for a_classes in as_classes]
    
    return as_ignore_classes, list(b_ignore_classes)


def _find_private_classes_when_sources_as_to_target_b(as_classes: List[List[str]], b_classes: List[str], da_mode):
    thres = {'da': 3, 'partial_da': 2, 'open_set_da': 1, 'universal_da': 0}[da_mode]
    
    from functools import reduce
    a_classes = reduce(lambda res, cur: res | set(cur), as_classes, set())
    
    if set(a_classes) == set(b_classes):
        # a is equal to b, normal
        # 1. no ignore classes; 2. match class idx
        a_private_classes, b_private_classes = [], []
    
    elif set(a_classes) > set(b_classes):
        # a contains b, partial
        a_private_classes, b_private_classes = [], []
        # if thres == 2 or thres == 0: # ignore extra classes in a
        #     a_private_classes = set(a_classes) - set(b_classes)
        # if thres == 0: # ignore extra classes in a
        #     a_private_classes = set(a_classes) - set(b_classes)
            
    elif set(a_classes) < set(b_classes):
        # a is contained by b, open set
        a_private_classes, b_private_classes = [], []
        if thres == 1 or thres == 0: # ignore extra classes in b
            b_private_classes = set(b_classes) - set(a_classes)
    
    elif len(set(a_classes) & set(b_classes)) > 0:
        a_private_classes, b_private_classes = [], []
        if thres == 0:
            # a_private_classes = set(a_classes) - (set(a_classes) & set(b_classes))
            
            b_private_classes = set(b_classes) - (set(a_classes) & set(b_classes))
        elif thres == 1:
            b_private_classes = set(b_classes) - (set(a_classes) & set(b_classes))
        elif thres == 2:
            # a_private_classes = set(a_classes) - (set(a_classes) & set(b_classes))
            pass
        
    else:
        return None # a has no intersection with b, none
    
    return list(b_private_classes)


class _ABDatasetMetaInfo:
    def __init__(self, name, classes, task_type, object_type, class_aliases, shift_type):
        self.name = name
        self.classes = classes
        self.class_aliases = class_aliases
        self.shift_type = shift_type
        self.task_type = task_type
        self.object_type = object_type
        
        
def _get_dist_shift_type_when_source_a_to_target_b(a: _ABDatasetMetaInfo, b: _ABDatasetMetaInfo):
    if b.shift_type is None:
        return 'Dataset Shifts'
    
    if a.name in b.shift_type.keys():
        return b.shift_type[a.name]
    
    mid_dataset_name = list(b.shift_type.keys())[0]
    mid_dataset_meta_info = _ABDatasetMetaInfo(mid_dataset_name, *static_dataset_registery[mid_dataset_name][1:])
    
    return _get_dist_shift_type_when_source_a_to_target_b(a, mid_dataset_meta_info) + ' + ' + list(b.shift_type.values())[0]

    
def _handle_all_datasets_v2(source_datasets: List[_ABDatasetMetaInfo], target_datasets: List[_ABDatasetMetaInfo], da_mode):
    
    # 1. merge the same meaning classes
    classes_info_of_all_datasets = {
        d.name: (d.classes, d.class_aliases)
        for d in source_datasets + target_datasets
    }
    final_classes_of_all_datasets, rename_map = _merge_the_same_meaning_classes(classes_info_of_all_datasets)
    all_datasets_classes = copy.deepcopy(final_classes_of_all_datasets)

    # print(all_datasets_known_classes)
    
    # 2. find ignored classes according to DA mode
    # source_datasets_ignore_classes, target_datasets_ignore_classes = {d.name: [] for d in source_datasets}, \
    #     {d.name: [] for d in target_datasets}
    # source_datasets_private_classes, target_datasets_private_classes = {d.name: [] for d in source_datasets}, \
    #     {d.name: [] for d in target_datasets}
    target_source_relationship_map = {td.name: {} for td in target_datasets}
    # source_target_relationship_map = {sd.name: [] for sd in source_datasets}
    
    # 1. construct target_source_relationship_map
    for sd in source_datasets:#sd和td使列表中每一个元素（类）的实例
        for td in target_datasets:
            sc = all_datasets_classes[sd.name]
            tc = all_datasets_classes[td.name]
            
            if len(set(sc) & set(tc)) == 0:#只保留有相似类别的源域和目标域
                continue
            
            target_source_relationship_map[td.name][sd.name] = _get_dist_shift_type_when_source_a_to_target_b(sd, td)
    
    # print(target_source_relationship_map)
    # exit()
    
    source_datasets_ignore_classes = {}
    for td_name, v1 in target_source_relationship_map.items():
        for sd_name, v2 in v1.items():
            source_datasets_ignore_classes[sd_name + '|' + td_name] = []
    target_datasets_ignore_classes = {d.name: [] for d in target_datasets}
    target_datasets_private_classes = {d.name: [] for d in target_datasets}
    # 保证对于每个目标域上的DA都符合给定的label shift
    # 所以不同目标域就算对应同一个源域，该源域也可能不相同
    
    for td_name, v1 in target_source_relationship_map.items():
        sd_names = list(v1.keys())
        
        sds_classes = [all_datasets_classes[sd_name] for sd_name in sd_names]
        td_classes = all_datasets_classes[td_name]
        ss_ignore_classes, t_ignore_classes = _find_ignore_classes_when_sources_as_to_target_b(sds_classes, td_classes, da_mode)#根据DA方式不同产生ignore_classes
        t_private_classes = _find_private_classes_when_sources_as_to_target_b(sds_classes, td_classes, da_mode)
        
        for sd_name, s_ignore_classes in zip(sd_names, ss_ignore_classes):
            source_datasets_ignore_classes[sd_name + '|' + td_name] = s_ignore_classes
        target_datasets_ignore_classes[td_name] = t_ignore_classes
        target_datasets_private_classes[td_name] = t_private_classes

    source_datasets_ignore_classes = {k: sorted(set(v), key=v.index) for k, v in source_datasets_ignore_classes.items()}
    target_datasets_ignore_classes = {k: sorted(set(v), key=v.index) for k, v in target_datasets_ignore_classes.items()}
    target_datasets_private_classes = {k: sorted(set(v), key=v.index) for k, v in target_datasets_private_classes.items()}
    
    # for k, v in source_datasets_ignore_classes.items():
    #     print(k, len(v))
    # print()
    # for k, v in target_datasets_ignore_classes.items():
    #     print(k, len(v))
    # print()
    # for k, v in target_datasets_private_classes.items():
    #     print(k, len(v))
    # print()
    
    # print(source_datasets_private_classes, target_datasets_private_classes)
    # 3. reparse classes idx
    # 3.1. agg all used classes
    # all_used_classes = []
    # all_datasets_private_class_idx_map = {}
    
    # source_datasets_classes_idx_map = {}
    # for td_name, v1 in target_source_relationship_map.items():
    #     for sd_name, v2 in v1.items():
    #         source_datasets_classes_idx_map[sd_name + '|' + td_name] = []
    # target_datasets_classes_idx_map = {}
    
    global_idx = 0
    all_used_classes_idx_map = {}
    # all_datasets_known_classes = {d: [] for d in final_classes_of_all_datasets.keys()}
    for dataset_name, classes in all_datasets_classes.items():
        if dataset_name not in target_datasets_ignore_classes.keys():
            ignore_classes = [0] * 100000
            for sn, sic in source_datasets_ignore_classes.items():
                if sn.startswith(dataset_name):
                    if len(sic) < len(ignore_classes):
                        ignore_classes = sic
        else:
            ignore_classes = target_datasets_ignore_classes[dataset_name]
        private_classes = [] \
            if dataset_name not in target_datasets_ignore_classes.keys() else target_datasets_private_classes[dataset_name]
        
        for c in classes:
            if c not in ignore_classes and c not in all_used_classes_idx_map.keys() and c not in private_classes:
                all_used_classes_idx_map[c] = global_idx
                global_idx += 1
                
    # print(all_used_classes_idx_map)
    
    # dataset_private_class_idx_offset = 0
    target_private_class_idx = global_idx
    target_datasets_private_class_idx = {d: None for d in target_datasets_private_classes.keys()}
    
    for dataset_name, classes in final_classes_of_all_datasets.items():
        if dataset_name not in target_datasets_private_classes.keys():
            continue
        
        # ignore_classes = target_datasets_ignore_classes[dataset_name]
        private_classes = target_datasets_private_classes[dataset_name]
        # private_classes = [] \
        #     if dataset_name in source_datasets_private_classes.keys() else target_datasets_private_classes[dataset_name]
        # for c in classes:
        #     if c not in ignore_classes and c not in all_used_classes_idx_map.keys() and c in private_classes:
        #         all_used_classes_idx_map[c] = global_idx + dataset_private_class_idx_offset
                
        if len(private_classes) > 0:
            # all_datasets_private_class_idx[dataset_name] = global_idx + dataset_private_class_idx_offset
            # dataset_private_class_idx_offset += 1
            # if dataset_name in source_datasets_private_classes.keys():
            #     if source_private_class_idx is None:
            #         source_private_class_idx = global_idx if target_private_class_idx is None else target_private_class_idx + 1
            #     all_datasets_private_class_idx[dataset_name] = source_private_class_idx
            # else:
            #     if target_private_class_idx is None:
            #         target_private_class_idx = global_idx if source_private_class_idx is None else source_private_class_idx + 1
            #     all_datasets_private_class_idx[dataset_name] = target_private_class_idx
            target_datasets_private_class_idx[dataset_name] = target_private_class_idx
            target_private_class_idx += 1
            
            
    # all_used_classes = sorted(set(all_used_classes), key=all_used_classes.index)
    # all_used_classes_idx_map = {c: i for i, c in enumerate(all_used_classes)}
    
    # print('rename_map', rename_map)
    
    # 3.2 raw_class -> rename_map[raw_classes] -> all_used_classes_idx_map
    all_datasets_e2e_idx_map = {}
    all_datasets_e2e_class_to_idx_map = {}
    
    for td_name, v1 in target_source_relationship_map.items():
        sd_names = list(v1.keys())
        sds_classes = [all_datasets_classes[sd_name] for sd_name in sd_names]
        td_classes = all_datasets_classes[td_name]
        
        for sd_name, sd_classes in zip(sd_names, sds_classes):
            cur_e2e_idx_map = {}
            cur_e2e_class_to_idx_map = {}
        
            for raw_ci, raw_c in enumerate(sd_classes):
                renamed_c = raw_c if raw_c not in rename_map[dataset_name] else rename_map[dataset_name][raw_c]
                
                ignore_classes = source_datasets_ignore_classes[sd_name + '|' + td_name]
                if renamed_c in ignore_classes:
                    continue
                
                idx = all_used_classes_idx_map[renamed_c]
                
                cur_e2e_idx_map[raw_ci] = idx
                cur_e2e_class_to_idx_map[raw_c] = idx
                
            all_datasets_e2e_idx_map[sd_name + '|' + td_name] = cur_e2e_idx_map
            all_datasets_e2e_class_to_idx_map[sd_name + '|' + td_name] = cur_e2e_class_to_idx_map   
        cur_e2e_idx_map = {}
        cur_e2e_class_to_idx_map = {}
        for raw_ci, raw_c in enumerate(td_classes):
            renamed_c = raw_c if raw_c not in rename_map[dataset_name] else rename_map[dataset_name][raw_c]
            
            ignore_classes = target_datasets_ignore_classes[td_name]
            if renamed_c in ignore_classes:
                continue
            
            if renamed_c in target_datasets_private_classes[td_name]:
                idx = target_datasets_private_class_idx[td_name]
            else:
                idx = all_used_classes_idx_map[renamed_c]
            
            cur_e2e_idx_map[raw_ci] = idx
            cur_e2e_class_to_idx_map[raw_c] = idx
            
        all_datasets_e2e_idx_map[td_name] = cur_e2e_idx_map
        all_datasets_e2e_class_to_idx_map[td_name] = cur_e2e_class_to_idx_map
        
    all_datasets_ignore_classes = {**source_datasets_ignore_classes, **target_datasets_ignore_classes}
    # all_datasets_private_classes = {**source_datasets_private_classes, **target_datasets_private_classes}
    
    classes_idx_set = []
    for d, m in all_datasets_e2e_class_to_idx_map.items():
        classes_idx_set += list(m.values())
    classes_idx_set = set(classes_idx_set)
    num_classes = len(classes_idx_set)

    return all_datasets_ignore_classes, target_datasets_private_classes, \
        all_datasets_e2e_idx_map, all_datasets_e2e_class_to_idx_map, target_datasets_private_class_idx, \
        target_source_relationship_map, rename_map, num_classes

    
def _build_scenario_info_v2(
    source_datasets_name: List[str],
    target_datasets_order: List[str],
    da_mode: str
):
    assert da_mode in ['close_set', 'partial', 'open_set', 'universal']
    da_mode = {'close_set': 'da', 'partial': 'partial_da', 'open_set': 'open_set_da', 'universal': 'universal_da'}[da_mode]
    
    source_datasets_meta_info = [_ABDatasetMetaInfo(d, *static_dataset_registery[d][1:]) for d in source_datasets_name]#获知对应的名字和对应属性，要添加数据集时，直接register就行
    target_datasets_meta_info = [_ABDatasetMetaInfo(d, *static_dataset_registery[d][1:]) for d in list(set(target_datasets_order))]

    all_datasets_ignore_classes, target_datasets_private_classes, \
        all_datasets_e2e_idx_map, all_datasets_e2e_class_to_idx_map, target_datasets_private_class_idx, \
        target_source_relationship_map, rename_map, num_classes \
        = _handle_all_datasets_v2(source_datasets_meta_info, target_datasets_meta_info, da_mode)
        
    return all_datasets_ignore_classes, target_datasets_private_classes, \
        all_datasets_e2e_idx_map, all_datasets_e2e_class_to_idx_map, target_datasets_private_class_idx, \
        target_source_relationship_map, rename_map, num_classes


def build_scenario_manually_v2(
    source_datasets_name: List[str],
    target_datasets_order: List[str],
    da_mode: str,
    data_dirs: Dict[str, str],
    transforms: Optional[Dict[str, Compose]] = None
):
    configs = copy.deepcopy(locals())#返回当前局部变量
    
    source_datasets_meta_info = [_ABDatasetMetaInfo(d, *static_dataset_registery[d][1:]) for d in source_datasets_name]
    target_datasets_meta_info = [_ABDatasetMetaInfo(d, *static_dataset_registery[d][1:]) for d in list(set(target_datasets_order))]

    all_datasets_ignore_classes, target_datasets_private_classes, \
        all_datasets_e2e_idx_map, all_datasets_e2e_class_to_idx_map, target_datasets_private_class_idx, \
        target_source_relationship_map, rename_map, num_classes \
        = _build_scenario_info_v2(source_datasets_name, target_datasets_order, da_mode)
    # from rich.console import Console
    # console = Console(width=10000)
    
    # def print_obj(_o):
    #     # import pprint
    #     # s = pprint.pformat(_o, width=140, compact=True)
    #     console.print(_o)
    
    # console.print('configs:', style='bold red')
    # print_obj(configs)
    # console.print('renamed classes:', style='bold red')
    # print_obj(rename_map)
    # console.print('discarded classes:', style='bold red')
    # print_obj(all_datasets_ignore_classes)
    # console.print('unknown classes:', style='bold red')
    # print_obj(target_datasets_private_classes)
    # console.print('class to index map:', style='bold red')
    # print_obj(all_datasets_e2e_class_to_idx_map)
    # console.print('index map:', style='bold red')
    # print_obj(all_datasets_e2e_idx_map)
    # console = Console()
    # # console.print('class distribution:', style='bold red')
    # # class_dist = {
    # #     k: {
    # #         '#known classes': len(all_datasets_known_classes[k]),
    # #         '#unknown classes': len(all_datasets_private_classes[k]),
    # #         '#discarded classes': len(all_datasets_ignore_classes[k])
    # #     } for k in all_datasets_ignore_classes.keys()
    # # }
    # # print_obj(class_dist)
    # console.print('corresponding sources of each target:', style='bold red')
    # print_obj(target_source_relationship_map)
    
    # return
    
    # res_source_datasets_map = {d: {split: get_dataset(d, data_dirs[d], split, getattr(transforms, d, None),
    #                                                   all_datasets_ignore_classes[d], all_datasets_e2e_idx_map[d]) 
    #                                for split in ['train', 'val', 'test']} 
    #                            for d in source_datasets_name}
    # res_target_datasets_map = {d: {'train': get_num_limited_dataset(get_dataset(d, data_dirs[d], 'test', getattr(transforms, d, None), 
    #                                                   all_datasets_ignore_classes[d], all_datasets_e2e_idx_map[d]), 
    #                                                                 num_samples_in_each_target_domain),
    #                                'test': get_dataset(d, data_dirs[d], 'test', getattr(transforms, d, None), 
    #                                                   all_datasets_ignore_classes[d], all_datasets_e2e_idx_map[d])
    #                                } 
    #                            for d in list(set(target_datasets_order))}
    
    # res_source_datasets_map = {d: {split: get_dataset(d.split('|')[0], data_dirs[d.split('|')[0]], split, 
    #                                                   getattr(transforms, d.split('|')[0], None),
    #                                                   all_datasets_ignore_classes[d], all_datasets_e2e_idx_map[d]) 
    #                                for split in ['train', 'val', 'test']} 
    #                            for d in all_datasets_ignore_classes.keys() if d.split('|')[0] in source_datasets_name}
    
    # from functools import reduce
    # res_offline_train_source_datasets_map = {}
    # res_offline_train_source_datasets_map_names = {}
    
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

    # res_target_datasets_map = {d: {split: get_dataset(d, data_dirs[d], split, getattr(transforms, d, None),
    #                                                   all_datasets_ignore_classes[d], all_datasets_e2e_idx_map[d]) 
    #                                for split in ['train', 'val', 'test']} 
    #                            for d in list(set(target_datasets_order))}
    
    from .scenario import Scenario, DatasetMetaInfo
    
    # test_scenario = Scenario(
    #     config=configs,
    #     offline_source_datasets_meta_info={
    #         d: DatasetMetaInfo(d, 
    #                            {k: v for k, v in all_datasets_e2e_class_to_idx_map[res_offline_train_source_datasets_map_names[d]].items()}, 
    #                            None)
    #         for d in source_datasets_name
    #     },
    #     offline_source_datasets={d: res_offline_train_source_datasets_map[d] for d in source_datasets_name},

    #     online_datasets_meta_info=[
    #         (
    #             {sd + '|' + d:  DatasetMetaInfo(d, 
    #                            {k: v for k, v in all_datasets_e2e_class_to_idx_map[sd + '|' + d].items()}, 
    #                            None)
    #              for sd in target_source_relationship_map[d].keys()},
    #             DatasetMetaInfo(d, 
    #                            {k: v for k, v in all_datasets_e2e_class_to_idx_map[d].items() if k not in target_datasets_private_classes[d]}, 
    #                            target_datasets_private_class_idx[d])
    #         )
    #         for d in target_datasets_order
    #     ],
    #     online_datasets={**res_source_datasets_map, **res_target_datasets_map},
    #     target_domains_order=target_datasets_order,
    #     target_source_map=target_source_relationship_map,
    #     num_classes=num_classes
    # )
    import os
    os.environ['_ZQL_NUMC'] = str(num_classes)

    test_scenario = Scenario(config=configs, all_datasets_ignore_classes_map=all_datasets_ignore_classes,
                             all_datasets_idx_map=all_datasets_e2e_idx_map,
                             target_domains_order=target_datasets_order,
                             target_source_map=target_source_relationship_map,
                             all_datasets_e2e_class_to_idx_map=all_datasets_e2e_class_to_idx_map,
                             num_classes=num_classes)


    return test_scenario


if __name__ == '__main__':
    test_scenario = build_scenario_manually_v2(['CIFAR10', 'SVHN'],
                               ['STL10', 'MNIST', 'STL10', 'USPS', 'MNIST', 'STL10'],
                               'close_set')
    print(test_scenario.num_classes)
    