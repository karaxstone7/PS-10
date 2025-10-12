import sys, os, time
sys.path.insert(0, '.')

from models.model import BaseNet

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel import gather

import dataset as myDataLoader
import Transforms as myTransforms
from metric_tool import ConfuseMatrixMeter
from PIL import Image
import numpy as np
from argparse import ArgumentParser


def BCEDiceLoss(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    return bce + 1 - dice


@torch.no_grad()
def val(args, val_loader, model, epoch):
    os.makedirs(args.vis_dir, exist_ok=True)
    model.eval()
    cm = ConfuseMatrixMeter(n_class=2)
    epoch_loss, total_batches = [], len(val_loader)
    print(total_batches)

    # best-effort way to access filenames, falls back to index
    def _name_for(i):
        try:
            return os.path.basename(val_loader.dataset.file_list[i])
        except Exception:
            return f"sample_{i}.png"

    for it, (img, target) in enumerate(val_loader):
        pre_img, post_img = img[:, 0:3], img[:, 3:6]
        t0 = time.time()

        if args.onGPU and torch.cuda.is_available():
            pre_img = pre_img.cuda(non_blocking=True)
            post_img = post_img.cuda(non_blocking=True)
            target  = target.cuda(non_blocking=True)

        pre_img_var  = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var   = torch.autograd.Variable(target).float()

        out  = model(pre_img_var, post_img_var)
        loss = BCEDiceLoss(out, target_var)
        pred = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out)).long()

        epoch_loss.append(loss.item())
        if args.onGPU and torch.cuda.device_count() > 1:
            pred = gather(pred, 0, dim=0)

        pr = pred[0, 0].detach().float().cpu().numpy()
        gt = target_var[0, 0].detach().float().cpu().numpy()

        f1 = cm.update_cm(pr, gt)

        # save a color-coded comparison map
        idx_tp = np.logical_and(pr == 1, gt == 1)
        idx_fp = np.logical_and(pr == 1, gt == 0)
        idx_tn = np.logical_and(pr == 0, gt == 0)
        idx_fn = np.logical_and(pr == 0, gt == 1)

        cmap = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
        cmap[idx_tp] = [255, 255, 255]  # TP -> white
        cmap[idx_fp] = [255,   0,   0]  # FP -> red
        cmap[idx_tn] = [  0,   0,   0]  # TN -> black
        cmap[idx_fn] = [  0, 255, 255]  # FN -> cyan

        out_name = _name_for(it)
        if not out_name.lower().endswith(".png"):
            out_name += ".png"
        Image.fromarray(cmap).save(os.path.join(args.vis_dir, out_name))

        if it % 5 == 0:
            print('\r[%d/%d] F1: %.6f loss: %.3f time: %.3f' %
                  (it, total_batches, f1, loss.item(), time.time() - t0), end='')

    avg_loss = sum(epoch_loss) / max(1, len(epoch_loss))
    scores = cm.get_scores()
    return avg_loss, scores


def _strip_module(sd):
    from collections import OrderedDict
    out = OrderedDict()
    for k, v in sd.items():
        nk = k.replace("module.", "", 1) if isinstance(k, str) and k.startswith("module.") else k
        out[nk] = v
    return out


def _load_weights(model, args):
    tried, cands = [], []

    # (1) explicit --weight
    if getattr(args, "weight", ""):
        # if it's an absolute/relative path that doesn't exist, also try basename within savedir
        cands.append(args.weight)
        if args.savedir:
            cands.append(os.path.join(args.savedir, os.path.basename(args.weight)))

    # (2) common fallbacks inside savedir
    if args.savedir:
        cands += [
            os.path.join(args.savedir, "best_model.pth"),
            os.path.join(args.savedir, "last_model.pth"),
            os.path.join(args.savedir, "checkpoint.pth.tar"),
        ]

    for path in cands:
        if not path or not os.path.isfile(path):
            tried.append(path)
            continue
        print(f"Trying to load weights: {path}")
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and any(k in sd for k in ("state_dict", "model")):
            sd = sd.get("state_dict", sd.get("model", sd))
        if isinstance(sd, dict):
            sd = _strip_module(sd)
            model.load_state_dict(sd, strict=False)
            print(f"Loaded weights from: {path}")
            return
        # if it's not a dict, it might already be state_dict (rare)
        try:
            model.load_state_dict(sd, strict=False)
            print(f"Loaded weights from: {path}")
            return
        except Exception as e:
            print(f"Failed to load from {path}: {e}")
            tried.append(path)

    raise FileNotFoundError("No usable weights found. Tried: " + " | ".join([str(t) for t in tried if t]))


def ValidateSegmentation(args):
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(2333)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2333)

    model = BaseNet(3, 1)

    # === IMPORTANT: do NOT rewrite args.savedir. Use it as provided. ===
    if not args.savedir:
        args.savedir = "./results"
    args.vis_dir = os.path.join(args.savedir, "predict")
    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    # resolve dataset path
    if isinstance(args.file_root, str) and os.path.isdir(args.file_root):
        needed = [os.path.join(args.file_root, d) for d in ("A", "B", "label")]
        if not all(os.path.isdir(d) for d in needed):
            raise TypeError(f"{args.file_root} must contain A/, B/, and label/ folders")
    else:
        raise TypeError(f'{args.file_root} is not a valid dataset folder')

    if args.onGPU and torch.cuda.is_available():
        model = model.cuda()

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters (excluding idr): ' + str(total_params))

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std  = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    val_tf = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    test_data = myDataLoader.Dataset("test", file_root=args.file_root, transform=val_tf)
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=False
    )

    if args.onGPU and torch.cuda.is_available():
        cudnn.benchmark = True

    logFileLoc = os.path.join(args.savedir, args.logFile)
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_params)))
        logger.write("\n%s\t%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Kappa', 'IoU', 'F1', 'R', 'P'))
    logger.flush()

    # Load weights (robust)
    _load_weights(model, args)

    # Eval
    loss_te, score_te = val(args, testLoader, model, 0)
    print("\nTest :\t Kappa (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f" %
          (score_te['Kappa'], score_te['IoU'], score_te['F1'], score_te['recall'], score_te['precision']))
    logger.write("\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" %
                 ('Test', score_te['Kappa'], score_te['IoU'], score_te['F1'],
                  score_te['recall'], score_te['precision']))
    logger.flush()
    logger.close()

    try:
        import scipy.io as scio
        scio.savemat(os.path.join(args.vis_dir, 'results.mat'), score_te)
    except Exception as e:
        print("[warn] could not save results.mat:", e)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default="", help='Folder containing A,B,label')
    parser.add_argument('--inWidth', type=int, default=256)
    parser.add_argument('--inHeight', type=int, default=256)
    parser.add_argument('--max_steps', type=int, default=40000)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--step_loss', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_mode', default='poly')
    parser.add_argument('--savedir', default='./results')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--logFile', default='testLog.txt')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--weight', default='', type=str, help='Path to weights or basename (best_model.pth) inside savedir')
    parser.add_argument('--ms', type=int, default=0)
    args = parser.parse_args()
    print('Called with args:')
    print(args)
    ValidateSegmentation(args)
