# /content/ScratchFormer/models/trainer.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import utils
from utils import de_norm
from models.networks import define_G, get_scheduler
from misc.logger_tool import Logger, Timer
from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy, FocalLoss, mIoULoss, mmIoULoss, get_alpha, softmax_helper
import models.losses as losses

class CDTrainer():
    def __init__(self, args, dataloaders):
        self.args = args
        self.dataloaders = dataloaders
        self.device = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() and len(args.gpu_ids) > 0 else "cpu")

        self.n_class = args.n_class
        self.batch_size = args.batch_size
        self.shuffle_AB = args.shuffle_AB
        self.multi_scale_train = args.multi_scale_train
        self.multi_scale_infer = args.multi_scale_infer
        self.weights = tuple(args.multi_pred_weights)
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        self.net_G = define_G(args=args, gpu_ids=[int(g) for g in args.gpu_ids if torch.cuda.device_count() > int(g)])

        if args.optimizer == "sgd":
            self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        elif args.optimizer == "adam":
            self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=args.lr)
        elif args.optimizer == "adamw":
            self.optimizer_G = optim.AdamW(self.net_G.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        if args.loss == 'ce':
            self._pxl_loss = cross_entropy
        elif args.loss == 'bce':
            self._pxl_loss = losses.binary_ce
        elif args.loss == 'fl':
            alpha = get_alpha(dataloaders['train'])
            self._pxl_loss = FocalLoss(apply_nonlin=softmax_helper, alpha=alpha, gamma=2, smooth=1e-5)
        elif args.loss == 'miou':
            alpha = np.asarray(get_alpha(dataloaders['train']))
            alpha = alpha / np.sum(alpha)
            weights = 1 - torch.from_numpy(alpha).to(self.device)
            self._pxl_loss = mIoULoss(weight=weights, size_average=True, n_classes=args.n_class).to(self.device)
        elif args.loss == 'mmiou':
            self._pxl_loss = mmIoULoss(n_classes=args.n_class).to(self.device)
        else:
            raise NotImplementedError(args.loss)

        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        self.logger = Logger(os.path.join(args.checkpoint_dir, 'log.txt'))
        self.logger.write_dict_str(args.__dict__)
        self.timer = Timer()

        self.epoch_to_start = 0
        self.epoch_id = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.max_num_epochs = args.max_epochs
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = self.max_num_epochs * self.steps_per_epoch

        self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy')) if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')) else np.array([], np.float32)
        self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy')) if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')) else np.array([], np.float32)

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

    def _update_metric(self):
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_final_pred.detach()

        if G_pred.shape[-2:] != target.shape[-2:]:
            G_pred = F.interpolate(G_pred, size=target.shape[-2:], mode='bilinear', align_corners=True)

        G_pred = torch.argmax(G_pred, dim=1).unsqueeze(1)
        if target.ndim == 4:
            target = target.squeeze(1)

        G_pred_np = G_pred.cpu().numpy()
        target_np = target.cpu().numpy()

        if G_pred_np.shape != target_np.shape:
            target_np = np.array([
                cv2.resize(t, (G_pred_np.shape[-1], G_pred_np.shape[-2]), interpolation=cv2.INTER_NEAREST)
                for t in target_np
            ])

        return self.running_metric.update_cm(pr=G_pred_np, gt=target_np)

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)
        self.G_final_pred = self.G_pred[-1] if not self.multi_scale_infer else sum(self.G_pred) / len(self.G_pred)

    def _backward_G(self):
        gt = self.batch['L'].to(self.device).float()
        if self.multi_scale_train:
            loss = 0
            for i, pred in enumerate(self.G_pred):
                tgt = gt if pred.size(2) == gt.size(2) else F.interpolate(gt, size=pred.size(2), mode='nearest')
                loss += self.weights[i] * self._pxl_loss(pred, tgt)
            self.G_loss = loss
        else:
            self.G_loss = self._pxl_loss(self.G_pred[-1], gt)
        self.G_loss.backward()

    def _save_checkpoint(self, name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.checkpoint_dir, name))

    def _load_checkpoint(self, name='last_ckpt.pt'):
        path = os.path.join(self.checkpoint_dir, name)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.net_G.load_state_dict(ckpt['model_G_state_dict'])
            self.optimizer_G.load_state_dict(ckpt['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(ckpt['exp_lr_scheduler_G_state_dict'])
            self.epoch_to_start = ckpt['epoch_id'] + 1
            self.best_val_acc = ckpt['best_val_acc']
            self.best_epoch_id = ckpt['best_epoch_id']
            self.net_G.to(self.device)
        elif self.args.pretrain is not None:
            self.net_G.load_state_dict(torch.load(self.args.pretrain), strict=False)
            self.net_G.to(self.device)

    def train_models(self):
        self._load_checkpoint()

        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
            self.logger.write(f"lr: {self.optimizer_G.param_groups[0]['lr']:.7f}\n\n")
            self.running_metric.clear()
            self.is_training = True
            self.net_G.train()
            for self.batch_id, batch in tqdm(enumerate(self.dataloaders['train']), total=len(self.dataloaders['train'])):
                self._forward_pass(batch)
                self.optimizer_G.zero_grad()
                self._backward_G()
                self.optimizer_G.step()
                self._update_metric()
            self._end_epoch(train=True)

            self.logger.write('Begin evaluation...\n')
            self.running_metric.clear()
            self.is_training = False
            self.net_G.eval()
            with torch.no_grad():
                for self.batch_id, batch in enumerate(self.dataloaders['val']):
                    self._forward_pass(batch)
                    self._update_metric()
            self._end_epoch(train=False)

            self._update_lr()
            self._save_checkpoint('last_ckpt.pt')
            if self.epoch_acc > self.best_val_acc:
                self.best_val_acc = self.epoch_acc
                self.best_epoch_id = self.epoch_id
                self._save_checkpoint('best_ckpt.pt')
                self.logger.write('*' * 10 + 'Best model updated!\n\n')

    def _end_epoch(self, train):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.logger.write(f"Is_training: {train}. Epoch {self.epoch_id}, mF1: {self.epoch_acc:.4f}\n")
        for k, v in scores.items():
            self.logger.write(f"{k}: {v:.5f} ")
        self.logger.write('\n\n')
        acc_arr = self.TRAIN_ACC if train else self.VAL_ACC
        acc_arr = np.append(acc_arr, [self.epoch_acc])
        fname = 'train_acc.npy' if train else 'val_acc.npy'
        np.save(os.path.join(self.checkpoint_dir, fname), acc_arr)
        if train:
            self.TRAIN_ACC = acc_arr
        else:
            self.VAL_ACC = acc_arr

    def _update_lr(self):
        self.exp_lr_scheduler_G.step()
