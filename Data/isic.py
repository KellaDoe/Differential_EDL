import pandas as pd
import numpy as np
import torch
import torchvision
import PIL
import csv
import os

from torch.utils.data import Dataset
from torchvision import transforms
from fastai import *
from fastai.vision import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ISICDataset(Dataset):
    def __init__(self, transform, test_all=0, data_mode='train', fold=0):
        all_image_path = []
        all_label = []
        with open('/mnt/Data/ISIC_2019/ISIC_2019_Training_GroundTruth.csv',encoding='utf-8') as labels_file:
            count = 0
            for row in csv.reader(labels_file):
                if(count == 0):
                    count += 1
                    continue
                filename = row.pop(0) + '.jpg'
                # print(filename)
                if os.path.exists('/mnt/Data/ISIC_2019/ISIC_2019_Training_Input' + '/' + filename):
                    all_image_path.append(filename)
                    all_label.append(np.array(row).astype("float"))

        if test_all == 1:
            self.in_id = [0,1,2,3,4,5,6,7]
            self.ood_id = []
        else:
            self.in_id = [0,1,2,3,4,7]
            self.ood_id = [5,6]

        self.transform = transform
        self.data_mode = data_mode
        self.test_all = test_all
        # 图片id
        self.paths = []
        # 图片标签
        self.labels = []
        if data_mode != 'ood':
            for index in range(len(all_image_path)):
                # for id_index in self.in_id:
                #     if all_label[index][id_index] == 1.0:
                #         self.paths.append(all_image_path[index])
                #         self.labels.append(all_label[index])
                if np.argmax(all_label[index]) in self.in_id:
                    self.paths.append(all_image_path[index])
                    self.labels.append(all_label[index])
            
            val_index = np.linspace(0, len(self.paths), len(self.paths) // 5, endpoint=False, dtype=np.int)
            train_index = np.setdiff1d(np.arange(len(self.paths)), val_index)
            if data_mode == 'train':
                self.paths = np.array(self.paths)[train_index]
                self.labels = np.array(self.labels)[train_index]
            else:
                self.paths = np.array(self.paths)[val_index]
                self.labels = np.array(self.labels)[val_index]
        else:
            for index in range(len(all_image_path)):
                for ood_index in self.ood_id:
                    if all_label[index][ood_index] == 1.0:
                        self.paths.append(all_image_path[index])
                        self.labels.append(all_label[index])

    def __getitem__(self, idx):
        # Convert image to tensor and pre-process using transform
        img = PIL.Image.open('/mnt/Data/ISIC_2019/ISIC_2019_Training_Input/'+self.paths[idx]).convert('RGB')
        img = self.transform(img)
        if self.test_all == 1:
            id_map = [0,1,2,3,4,5,6,7]
        else:
            id_map = [0,1,2,3,4,0,0,5,0]
        # Convert caption to tensor of word ids.
        # target = torch.tensor(self.labels[idx], dtype=torch.int64)
        # return pre-processed image and caption tensor
        if self.data_mode == 'ood':
            target = torch.zeros(6, dtype=torch.int64)
        else:
            target = torch.tensor(self.labels[idx], dtype=torch.int64)
            # print(id_map[torch.argmax(target)])
            target = torch.nn.functional.one_hot(torch.tensor(id_map[torch.argmax(target)], dtype=torch.int64), num_classes=len(self.in_id))
        return img, target

    def __len__(self):
        return len(self.paths)