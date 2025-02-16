from ..data_aug import imagenet_like_image_train_aug, imagenet_like_image_test_aug
from ..ab_dataset import ABDataset
from ..dataset_split import train_val_split
from ..dataset_cache import get_dataset_cache_path, read_cached_dataset_status, cache_dataset_status
from torchvision.datasets import ImageFolder
import os
from typing import Dict, List, Optional
from torchvision.transforms import Compose

from ..registery import dataset_register


with open(os.path.join(os.path.dirname(__file__), 'imagenet_classes.txt'), 'r') as f:
    classes = [line.split(' ')[2].strip() for line in f.readlines()]
    assert len(classes) == 1000
    
@dataset_register(
    name='ImageNet', 
    classes=classes, 
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class ImageNet(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = imagenet_like_image_train_aug() if split == 'train' else imagenet_like_image_test_aug()
            self.transform = transform
        root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        dataset = ImageFolder(root_dir, transform=transform)
        
        cache_file_path = get_dataset_cache_path(root_dir, classes, ignore_classes, idx_map)
        if os.path.exists(cache_file_path):
            dataset.samples = read_cached_dataset_status(cache_file_path, 'ImageNet-' + split)
        else:
            if len(ignore_classes) > 0:
                ignore_classes_idx = [classes.index(c) for c in ignore_classes]
                dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]    
            if idx_map is not None:
                dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]
            
            cache_dataset_status(dataset.samples, cache_file_path, 'ImageNet-' + split)
        
        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset
