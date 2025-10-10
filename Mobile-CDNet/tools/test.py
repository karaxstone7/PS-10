import sys, os, time
sys.path.insert(0, '.')

from models.model import BaseNet

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel import gather
import torch.optim.lr_scheduler

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


def BCE(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    return bce


@torch.no_grad()
def val(args, val_loader, model, epoch):
    model.eval()
    salEvalVal = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []

    total_batches = len(val_loader)
    print(total_batches)
    for iter, batched_inputs in enumerate(val_loader):
        img, target = batched_inputs
        # try to get a sensible filename
        try:
            img_name = val_loader.sampler.data_source.file_list[iter]
        except Exception:
            img_name = f"sample_{iter}.png"

        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        start_time = time.time()

        if args.onGPU:
            pre_img = pre_img.cuda(non_blocking=True)
            post_img = post_img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()

        # run the model
        output = model(pre_img_var, post_img_var)
        loss = BCEDiceLoss(output, target_var)

        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

        time_taken = time.time() - start_time
        epoch_loss.append(loss.item())

        # multi-gpu gather if needed
        if args.onGPU and torch.cuda.device_count() > 1:
            pred = gather(pred, 0, dim=0)

        # save RGB-coded change map (TP=white, FP=red, TN=black, FN=cyan)
        pr = pred[0, 0].detach().cpu().numpy()
        gt = target_var[0, 0].detach().cpu().numpy()

        index_tp = np.where(np.logical_and(pr == 1, gt == 1))
        index_fp = np.where(np.logical_and(pr == 1, gt == 0))
        index_tn = np.where(np.logical_and(pr == 0, gt == 0))
        index_fn = np.where(np.logical_and(pr == 0, gt == 1))

        cmap = np.zeros([gt.shape[0], gt.shape[1], 3], dtype=np.uint8)
        cmap[index_tp] = [255, 255, 255]  # white
        cmap[index_fp] = [255, 0, 0]      # red
        cmap[index_tn] = [0, 0, 0]        # black
        cmap[index_fn] = [0, 255, 255]    # cyan

        out_name = os.path.basename(str(img_name))
        if not out_name.lower().endswith(".png"):
            out_name += ".png"
        Image.fromarray(cmap).save(os.path.join(args.vis_dir, out_name))

        f1 = salEvalVal.update_cm(pr, gt)

        if iter % 5 == 0:
            print('\r[%d/%d] F1: %.6f loss: %.3f time: %.3f' %
                  (iter, total_batches, f1, loss.item(), time_taken), end='')

    average_epoch_loss_val = sum(epoch_loss) / max(1, len(epoch_loss))
    scores = salEvalVal.get_scores()
    return average_epoch_loss_val, scores


def _strip_module(sd):
    from collections import OrderedDict
    new = OrderedDict()
    for k, v in sd.items():
        nk = k.replace("module.", "", 1) if isinstance(k, str) and k.startswith("module.") else k
        new[nk] = v
    return new


def _load_weights(model, args):
    tried = []
    # 1) prefer explicit --weight
    if getattr(args, "weight", "") and os.path.isfile(args.weight):
        tried.append(args.weight)
        sd = torch.load(args.weight, map_location="cpu")
        if isinstance(sd, dict) and any(k in sd for k in ("state_dict", "model")):
            sd = sd.get("state_dict", sd.get("model", sd))
        if isinstance(sd, dict):
            sd = _strip_module(sd)
        model.load_state_dict(sd, strict=False)
        print(f"Loaded weights from --weight: {args.weight}")
        return

    # 2) fallback to best_model.pth in savedir (legacy behavior)
    fallback = os.path.join(args.savedir, 'best_model.pth')
    if os.path.isfile(fallback):
        tried.append(fallback)
        sd = torch.load(fallback, map_location="cpu")
        if isinstance(sd, dict) and any(k in sd for k in ("state_dict", "model")):
            sd = sd.get("state_dict", sd.get("model", sd))
        if isinstance(sd, dict):
            sd = _strip_module(sd)
        model.load_state_dict(sd, strict=False)
        print(f"Loaded fallback weights: {fallback}")
        return

    raise FileNotFoundError("No usable weights found. Tried: " + " | ".join(tried))


def ValidateSegmentation(args):
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    model = BaseNet(3, 1)

    # Build output dirs
    split_name = os.path.basename(os.path.normpath(str(args.file_root)))
    args.savedir = os.path.join(
        args.savedir + '_' + split_name + '_iter_' + str(args.max_steps) + '_lr_' + str(args.lr)
    )
    args.vis_dir = os.path.join(args.savedir, "predict")
    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    # Accept either known aliases or a real path with A/B/label
    if isinstance(args.file_root, str) and os.path.isdir(args.file_root):
        needed = [os.path.join(args.file_root, d) for d in ("A", "B", "label")]
        if not all(os.path.isdir(d) for d in needed):
            raise TypeError(f"{args.file_root} must contain A/, B/, and label/ folders")
    elif args.file_root == 'LEVIR':
        args.file_root = 'H:\\penghaifeng\\LEVIR-CD'
    elif args.file_root == 'BCDD':
        args.file_root = 'H:\\penghaifeng\\BCDD'
    elif args.file_root == 'SYSU':
        args.file_root = 'H:\\penghaifeng\\SYSU-CD'
    elif args.file_root == 'CDD':
        args.file_root = '/home/guan/Documents/Datasets/ChangeDetection/CDD'
    elif args.file_root == 'testLEVIR':
        args.file_root = '../samples'
    else:
        # final fallback: if it's not a dir, error clearly
        raise TypeError('%s has not defined' % args.file_root)

    if args.onGPU:
        model = model.cuda()

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters (excluding idr): ' + str(total_params))

    # IMPORTANT: keep same normalization you used for training
    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std  = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    test_data = myDataLoader.Dataset("test", file_root=args.file_root, transform=valDataset)
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    if args.onGPU:
        cudnn.benchmark = True

    logFileLoc = os.path.join(args.savedir, args.logFile)
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_params)))
        logger.write("\n%s\t%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Kappa', 'IoU', 'F1', 'R', 'P'))
    logger.flush()

    # ---- Load weights
    _load_weights(model, args)

    # ---- Eval
    loss_test, score_test = val(args, testLoader, model, 0)
    print("\nTest :\t Kappa (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f" %
          (score_test['Kappa'], score_test['IoU'], score_test['F1'],
           score_test['recall'], score_test['precision']))
    logger.write("\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (
        'Test', score_test['Kappa'], score_test['IoU'], score_test['F1'],
        score_test['recall'], score_test['precision']))
    logger.flush()
    logger.close()

    import scipy.io as scio
    scio.savemat(os.path.join(args.vis_dir, 'results.mat'), score_test)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default="", help='Either a dataset alias (LEVIR/BCDD/SYSU/CDD) or a folder with A,B,label')
    parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
    parser.add_argument('--max_steps', type=int, default=40000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=3, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy, step or poly')
    parser.add_argument('--savedir', default='./results', help='Directory to save the results')
    parser.add_argument('--resume', default=None, help='(unused here) checkpoint to continue training')
    parser.add_argument('--logFile', default='testLog.txt', help='File that stores the logs')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weight', default='', type=str, help='Path to pretrained weights (.pth or .tar)')
    parser.add_argument('--ms', type=int, default=0, help='apply multi-scale training, default False')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    ValidateSegmentation(args)
a
