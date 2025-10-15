import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import cv2
import torch
import torch.nn.functional as F

from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils


class CDEvaluator():
    def __init__(self, args, dataloader):
        self.dataloader = dataloader
        self.n_class = args.n_class

        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device(
            "cuda:%s" % args.gpu_ids[0]
            if torch.cuda.is_available() and len(args.gpu_ids) > 0
            else "cpu"
        )
        print(self.device)

        # metrics
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # logger
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        # training stats
        self.epoch_acc = 0.0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # ensure dirs
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):
        ckpt_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'no such checkpoint {checkpoint_name} at {ckpt_path}')

        self.logger.write(f'loading checkpoint: {ckpt_path}\n')

        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location='cpu')

        if isinstance(checkpoint, dict):
            if 'model_G_state_dict' in checkpoint:
                state_dict = checkpoint['model_G_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        def strip_prefix(k):
            for pref in ('module.', 'model.', 'netG.', 'G.'):
                if k.startswith(pref):
                    return k[len(pref):]
            return k

        state_dict_clean = OrderedDict((strip_prefix(k), v) for k, v in state_dict.items())

        if isinstance(self.net_G, torch.nn.DataParallel):
            self.net_G = self.net_G.module

        missing, unexpected = self.net_G.load_state_dict(state_dict_clean, strict=False)
        self.net_G.to(self.device)
        self.net_G.eval()

        if isinstance(checkpoint, dict):
            self.best_val_acc = checkpoint.get('best_val_acc', self.best_val_acc)
            self.best_epoch_id = checkpoint.get('best_epoch_id', self.best_epoch_id)

        self.logger.write(f'Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}\n')
        self.logger.write(
            'Eval Historical_best_acc = %.4f (at epoch %d)\n\n'
            % (self.best_val_acc, self.best_epoch_id)
        )

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _update_metric(self):
        """update metric"""
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(
            pr=G_pred.cpu().numpy(), gt=target.cpu().numpy()
        )
        return current_score

    def _collect_running_batch_states(self):
        """Visual logging for every batch"""
        running_acc = self._update_metric()
        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            msg = f'Is_training: {self.is_training}. [{self.batch_id},{m}], running_mf1: {running_acc:.5f}\n'
            self.logger.write(msg)

        vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
        vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))
        vis_pred = utils.make_numpy_grid(self._visualize_pred())
        vis_gt = utils.make_numpy_grid(self.batch['L'])

        # --- ensure all visuals are 3‑channel for stacking ---
        def ensure_rgb(arr):
            if arr.ndim == 2:
                return np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] != 3:
                return np.stack([arr.squeeze()] * 3, axis=-1)
            return arr

        vis_input = ensure_rgb(vis_input)
        vis_input2 = ensure_rgb(vis_input2)
        vis_pred = ensure_rgb(vis_pred)
        vis_gt = ensure_rgb(vis_gt)

        if vis_gt.shape != vis_input.shape:
            w, h = vis_gt.shape[1], vis_gt.shape[0]
            vis_input = cv2.resize(vis_input, (w, h), interpolation=cv2.INTER_LINEAR)
            vis_input2 = cv2.resize(vis_input2, (w, h), interpolation=cv2.INTER_LINEAR)
        if vis_gt.shape != vis_pred.shape:
            w, h = vis_gt.shape[1], vis_gt.shape[0]
            vis_pred = cv2.resize(vis_pred, (w, h), interpolation=cv2.INTER_NEAREST)

        vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        vis = np.clip(vis, 0.0, 1.0)
        file_name = os.path.join(self.vis_dir, f"eval_{self.batch_id}.jpg")
        plt.imsave(file_name, vis)

    def _collect_epoch_states(self):
        scores_dict = self.running_metric.get_scores()
        np.save(os.path.join(self.checkpoint_dir, "scores_dict.npy"), scores_dict)
        self.epoch_acc = scores_dict.get("mf1", 0.0)

        with open(os.path.join(self.checkpoint_dir, f"{self.epoch_acc}.txt"), mode="a"):
            pass

        msg = " ".join([f"{k}: {v:.5f}" for k, v in scores_dict.items()])
        self.logger.write(msg + "\n\n")

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        """Forward pass + shape alignment for GT and prediction"""
        self.batch = batch
        img_in1 = batch["A"].to(self.device)
        img_in2 = batch["B"].to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)[-1]

        # resize prediction to 256×256
        self.G_pred = F.interpolate(
            self.G_pred, size=[256, 256], mode="bilinear", align_corners=False
        )

        # safely resize GT to match
        gt = batch["L"]
        if gt.ndim == 3:
            gt = gt.unsqueeze(1).float()  # [B,1,H,W]
        elif gt.ndim == 4 and gt.shape[1] == 1:
            gt = gt.float()
        else:
            raise ValueError(f"Unexpected GT shape: {gt.shape}")

        gt_resized = F.interpolate(gt, size=[256, 256], mode="nearest")
        self.batch["L"] = gt_resized.squeeze(1).long()  # [B,H,W]

    def eval_models(self, checkpoint_name="best_ckpt.pt"):
        self._load_checkpoint(checkpoint_name)
        self.logger.write("Begin evaluation...\n")
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()

        self._collect_epoch_states()
