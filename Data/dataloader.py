import torch
import numpy as np
import os
import sys
import random
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets

from torch.utils.data import sampler
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torch.utils.data import DataLoader


def data_class(dataset,test_all=0):
    if test_all:
        if dataset == 'isic': return 8
        elif dataset == 'bone': return 21
        elif dataset == 'retinal': return 8
        elif dataset == 'limuc': return 4
        elif dataset == 'dra': return 5
        elif dataset == 'pancreas_old': return 9
        elif dataset == 'pancreas128': return 8
        elif dataset == 'pancreas256': return 8
        elif dataset == 'pancreas256m': return 8
        elif dataset == 'pancreas_pdacv': return 8
        else:
            raise NotImplementedError('dataset not found')
    else:
        if dataset == 'isic': return 6
        elif dataset == 'bone': return 10
        elif dataset == 'retinal': return 7
        elif dataset == 'limuc': return 3
        elif dataset == 'dra': return 4
        elif dataset == 'pancreas_old': return 3
        elif dataset == 'pancreas128': return 6
        elif dataset == 'pancreas256': return 3
        elif dataset == 'pancreas256m': return 3
        elif dataset == 'pancreas_pdacv': return 7
        else: raise NotImplementedError('dataset not found')

def dataloader(dataset = 'isic', test_all=0, data_mode = 'train', batch_size = 32, num_workers = 8,
               image_size = 224, mean =(0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)):
    mean = mean
    std = std
    train_transform = transforms.Compose([
            # transforms.Resize((args.img_size, args.img_size)),
            transforms.Resize((256, 256)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    transform = train_transform if data_mode == 'train' else test_transform
    # 设置随机数种子
    torch.manual_seed(42)
    if dataset == 'isic':
        from Data.isic import ISICDataset
        valid = ISICDataset(transform=transform,test_all=test_all,data_mode=data_mode)
    elif dataset == 'bone':
        from Data.bone import BoneDataset
        valid = BoneDataset(transform=transform,test_all=test_all,data_mode=data_mode)
    elif dataset == 'retinal':
        from Data.retinal import RetinalDataset
        valid = RetinalDataset(transform=transform,test_all=test_all,data_mode=data_mode)
    elif dataset == 'limuc':
        from Data.limuc import LIMUCDataset
        valid = LIMUCDataset(transform=transform,test_all=test_all,data_mode=data_mode)
    elif dataset == 'dra':
        from Data.dra import DRADataset
        valid = DRADataset(transform=transform,test_all=test_all,data_mode=data_mode)
    elif dataset == 'pancreas_old':
        from Data.pancreas_old import PancreasDataset
        valid = PancreasDataset(transform=transform,test_all=test_all,data_mode=data_mode)
    elif dataset == 'pancreas128':
        from Data.pancreas_128 import PancreasDataset
        valid = PancreasDataset(transform=transform,test_all=test_all,data_mode=data_mode)
    elif dataset == 'pancreas256':
        from Data.pancreas_256 import PancreasDataset
        valid = PancreasDataset(transform=transform,test_all=test_all,data_mode=data_mode)
    elif dataset == 'pancreas256m':
        from Data.pancreas_256m import PancreasDataset
        valid = PancreasDataset(transform=transform,test_all=test_all,data_mode=data_mode)
    elif dataset == 'pancreas_pdacv':
        from Data.pancreas_pdacv import PancreasDataset
        valid = PancreasDataset(transform=transform,test_all=test_all,data_mode=data_mode)
    else:
        raise NotImplementedError('dataset not found')
    
    print(len(valid))
    valid_dl = DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return valid_dl