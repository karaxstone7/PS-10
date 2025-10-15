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

        # training stats (also used in eval logs)
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
        """
        Load a checkpoint into self.net_G robustly.
        Accepts files saved as:
          - {'model_G_state_dict': ...}
          - {'state_dict': ...}
          - raw state_dict dict
        Also strips common prefixes: 'module.', 'model.', 'netG.', 'G.'.
        """
        ckpt_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'no such checkpoint {checkpoint_name} at {ckpt_path}')

        self.logger.write(f'loading checkpoint: {ckpt_path}\n')

        # torch.load with compatibility for newer PyTorch
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location='cpu')

        # extract a state_dict
        if isinstance(checkpoint, dict):
            if 'model_G_state_dict' in checkpoint:
                state_dict = checkpoint['model_G_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # strip common prefixes
        def strip_prefix(k):
            for pref in ('module.', 'model.', 'netG.', 'G.'):
                if k.startswith(pref):
                    return k[len(pref):]
            return k

        state_dict_clean = OrderedDict((strip_prefix(k), v) for k, v in state_dict.items())

        # if wrapped in DataParallel, unwrap for loading
        if isinstance(self.net_G, torch.nn.DataParallel):
            self.net_G = self.net_G.module

        missing, unexpected = self.net_G.load_state_dict(state_dict_clean, strict=False)
        self.net_G.to(self.device)
        self.net_G.eval()

        # optional historical stats
        if isinstance(checkpoint, dict):
            self.best_val_acc = checkpoint.get('best_val_acc', self.best_val_acc)
            self.best_epoch_id = checkpoint.get('best_epoch_id', self.best_epoch_id)

        self.logger.write(
            f'Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}\n'
        )
        self.logger.write(
            'Eval Historical_best_acc = %.4f (at epoch %d)\n\n'
            % (self.best_val_acc, self.best_epoch_id)
        )

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(
            pr=G_pred.cpu().numpy(), gt=target.cpu().numpy()
        )
        return current_score

    def _collect_running_batch_states(self):
        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' % \
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        # Visuals
        vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
        vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))
        vis_pred = utils.make_numpy_grid(self._visualize_pred())
        vis_gt = utils.make_numpy_grid(self.batch['L'])

        if vis_gt.shape != vis_input.shape:
            w, h = vis_gt.shape[1], vis_gt.shape[0]
            vis_input = cv2.resize(vis_input, (w, h), interpolation=cv2.INTER_LINEAR)
            vis_input2 = cv2.resize(vis_input2, (w, h), interpolation=cv2.INTER_LINEAR)

        if vis_gt.shape != vis_pred.shape:
            w, h = vis_gt.shape[1], vis_gt.shape[0]
            vis_pred = cv2.resize(vis_pred, (w, h), interpolation=cv2.INTER_NEAREST)

        vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(self.vis_dir, f'eval_{self.batch_id}.jpg')
        plt.imsave(file_name, vis)

    def _collect_epoch_states(self):
        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)
        self.epoch_acc = scores_dict.get('mf1', 0.0)

        # drop a marker file for this mf1 score
        with open(os.path.join(self.checkpoint_dir, f'{self.epoch_acc}.txt'), mode='a'):
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)
        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)[-1]

        # resize prediction back to 256x256
        self.G_pred = F.interpolate(self.G_pred, size=[256, 256], mode='bilinear', align_corners=False)

    def eval_models(self, checkpoint_name='best_ckpt.pt'):
        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()

        self._collect_epoch_states()
