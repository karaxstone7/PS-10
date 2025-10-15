from argparse import ArgumentParser
import os
import torch
from models.evaluator import *
import utils
import numpy as np

print(torch.cuda.is_available())

"""
Final fixed evaluator runner for ScratchFormer
Ensures all visual outputs are 3-channel RGB and no mismatched dims.
"""

def main():
    # -----------------
    # Args
    # -----------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0 or 0,1,2. use -1 for CPU')
    parser.add_argument('--project_name', default='scratchformer', type=str)
    parser.add_argument('--print_models', default=False, type=bool, help='print models')
    parser.add_argument('--checkpoints_root', default='./checkpoints', type=str)
    parser.add_argument('--vis_root', default='vis', type=str)

    # Data
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default="test", type=str)
    parser.add_argument('--img_size', default=256, type=int)

    # Model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--net_G', default='ScratchFormer', type=str)
    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)

    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)

    # Paths
    args.checkpoint_dir = os.path.join(args.checkpoints_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    # -----------------
    # Data + Model
    # -----------------
    dataloader = utils.get_loader(
        args.data_name,
        img_size=args.img_size,
        batch_size=args.batch_size,
        is_train=False,
        split=args.split
    )

    model = CDEvaluator(args=args, dataloader=dataloader)

    # -----------------
    # Patch visualization helper dynamically
    # -----------------
    import matplotlib.pyplot as plt
    import cv2

    def safe_collect_running_batch_states(self):
        running_acc = self._update_metric()
        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            msg = f'Is_training: {self.is_training}. [{self.batch_id},{m}], running_mf1: {running_acc:.5f}\n'
            self.logger.write(msg)

        vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
        vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))
        vis_pred = utils.make_numpy_grid(self._visualize_pred())
        vis_gt = utils.make_numpy_grid(self.batch['L'])

        # --- unify dims to 3-channel ---
        def ensure_rgb(arr):
            arr = np.squeeze(arr)
            if arr.ndim == 2:
                return np.stack([arr] * 3, axis=-1)
            if arr.ndim == 3 and arr.shape[2] != 3:
                return np.stack([arr[..., 0]] * 3, axis=-1)
            return arr

        vis_input = ensure_rgb(vis_input)
        vis_input2 = ensure_rgb(vis_input2)
        vis_pred = ensure_rgb(vis_pred)
        vis_gt = ensure_rgb(vis_gt)

        h, w = vis_gt.shape[:2]
        vis_input = cv2.resize(vis_input, (w, h), interpolation=cv2.INTER_LINEAR)
        vis_input2 = cv2.resize(vis_input2, (w, h), interpolation=cv2.INTER_LINEAR)
        vis_pred = cv2.resize(vis_pred, (w, h), interpolation=cv2.INTER_NEAREST)

        vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        vis = np.clip(vis, 0.0, 1.0)
        file_name = os.path.join(self.vis_dir, f"eval_{self.batch_id}.jpg")
        plt.imsave(file_name, vis)

    # Replace the old method at runtime (no re-import required)
    from types import MethodType
    model._collect_running_batch_states = MethodType(safe_collect_running_batch_states, model)

    # -----------------
    # Run evaluation
    # -----------------
    model.eval_models(checkpoint_name=args.checkpoint_name)


if __name__ == '__main__':
    main()
