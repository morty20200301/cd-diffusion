from io import BytesIO
import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import data.util as Util
import os.path
import torchvision.transforms as standard_transforms

import numpy as np

IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "label"
label_suffix = ".png"

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.str_)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list

def get_img_post_path(root_dir,img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)


def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)


def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name) #.replace('.jpg', label_suffix)


def get_standard_transformations():
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

class CDDataset(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=-1):
        
        self.res = resolution
        self.data_len = data_len
        self.split = split

        self.root_dir = dataroot
        self.std_trans = get_standard_transformations()
        self.split = split
        self.img_name_list = []
        split = ["train", "val", "test"]

        if split == "train":
            for item in split:
                self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, item+'.txt')
                self.img_name_list += load_img_name_list(self.list_path)
        else:
            self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split + '.txt')
            self.img_name_list = load_img_name_list(self.list_path)

        self.dataset_len = len(self.img_name_list)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.data_len])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.data_len])

        img_A  = self.std_trans(Image.open(A_path).convert("RGB"))
        img_B  = self.std_trans(Image.open(B_path).convert("RGB"))
        
        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.data_len])
        img_lbl = Image.open(L_path)
        mask = np.array(img_lbl)
        mask[mask > 0] = 1
        mask = torch.from_numpy(mask)
        
        return {'A': img_A, 'B': img_B, 'L': mask, 'Index': index}
