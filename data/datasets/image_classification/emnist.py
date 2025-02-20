import enum
from ..data_aug import one_d_image_test_aug, one_d_image_train_aug
from ..ab_dataset import ABDataset
from ..dataset_split import train_val_split
from torchvision.datasets import EMNIST as RawEMNIST
import string
import numpy as np
from typing import Dict, List, Optional
from torchvision.transforms import Compose

from ..registery import dataset_register


@dataset_register(
    name='EMNIST', 
    classes=list(string.digits + string.ascii_letters), 
    class_aliases=[],
    task_type='Image Classification',
    object_type='Digit and Letter',
    shift_type=None
)
class EMNIST(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = one_d_image_train_aug() if split == 'train' else one_d_image_test_aug()
            self.transform = transform
        dataset = RawEMNIST(root_dir, 'byclass', train=split != 'test', transform=transform, download=True)
        
        dataset.targets = np.asarray(dataset.targets)

        if len(ignore_classes) > 0: 
            for ignore_class in ignore_classes:
                dataset.data = dataset.data[dataset.targets != classes.index(ignore_class)]
                dataset.targets = dataset.targets[dataset.targets != classes.index(ignore_class)]
        
        if idx_map is not None:
            # note: the code below seems correct but has bug!
            # for old_idx, new_idx in idx_map.items():
            #     dataset.targets[dataset.targets == old_idx] = new_idx
                
            for ti, t in enumerate(dataset.targets):
                dataset.targets[ti] = idx_map[t]
        
        if split != 'test':
            dataset = train_val_split(dataset, split)    
        return dataset
