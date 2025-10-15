import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils as tv_utils

import sys
import os
sys.path.append(os.path.abspath("."))  # Add ScratchFormer root to path

from datasets.CD_dataset import CDDataset  # ✅ keep as-is after step 2


def get_loader(data_name, img_size=256, batch_size=8, split='test',
               is_train=False, dataset='CDDataset'):
    root_dir = "/content/drive/MyDrive/datasets/LEVIR"
    label_transform = "norm"
    
    data_set = CDDataset(root_dir=root_dir, split=split,
                         img_size=img_size, is_train=is_train,
                         label_transform=label_transform)

    dataloader = DataLoader(data_set, batch_size=batch_size,
                            shuffle=is_train, num_workers=4)
    return dataloader

def get_loaders(args):
    root_dir = "/content/drive/MyDrive/datasets/LEVIR"
    label_transform = "norm"
    
    training_set = CDDataset(root_dir=root_dir, split=args.split,
                             img_size=args.img_size, is_train=True,
                             label_transform=label_transform)

    val_set = CDDataset(root_dir=root_dir, split=args.split_val,
                        img_size=args.img_size, is_train=False,
                        label_transform=label_transform)

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}
    return dataloaders

def make_numpy_grid(tensor_data, pad_value=0, padding=0):
    tensor_data = tensor_data.detach()
    vis = tv_utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis

def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5

def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])
