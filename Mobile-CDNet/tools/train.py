import sys, os, time, numpy as np
sys.path.insert(0, '.')
from argparse import ArgumentParser

from models.model import BaseNet

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel import gather
import torch.optim.lr_scheduler

import dataset as myDataLoader
import Transforms as myTransforms
from metric_tool import ConfuseMatrixMeter
import utils
import matplotlib.pyplot as plt


def BCEDiceLoss(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    return bce + 1 - dice


def BCE(inputs, targets):
    return F.binary_cross_entropy(inputs, targets)


@torch.no_grad()
def val(args, val_loader, model, epoch):
    os.makedirs(args.vis_dir, exist_ok=True)  # ensure vis dir exists
    model.eval()
    salEvalVal = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []
    total_batches = len(val_loader)
    print(total_batches)
    for iter, batched_inputs in enumerate(val_loader):
        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]
        start_time = time.time()

        if args.onGPU and torch.cuda.is_available():
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()

        output = model(pre_img_var, post_img_var)
        loss = BCEDiceLoss(output, target_var)
        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

        time_taken = time.time() - start_time
        epoch_loss.append(loss.item())

        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(pred, 0, dim=0)

        f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=target_var.cpu().numpy())
        if iter % 5 == 0:
            print('\r[%d/%d] F1: %3f loss: %.3f time: %.3f' %
                  (iter, total_batches, f1, loss.item(), time_taken), end='')

        if np.mod(iter, 200) == 1:
            vis_input  = utils.make_numpy_grid(utils.de_norm(pre_img_var[0:8]))
            vis_input2 = utils.make_numpy_grid(utils.de_norm(post_img_var[0:8]))
            vis_pred   = utils.make_numpy_grid(pred[0:8])
            vis_gt     = utils.make_numpy_grid(target_var[0:8])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(args.vis_dir, f'val_{epoch}_{iter}.jpg')
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            plt.imsave(file_name, vis)

    average_epoch_loss_val = sum(epoch_loss) / max(1, len(epoch_loss))
    scores = salEvalVal.get_scores()
    return average_epoch_loss_val, scores


def train(args, train_loader, model, optimizer, epoch, max_batches, cur_iter=0, lr_factor=1.):
    os.makedirs(args.vis_dir, exist_ok=True)  # ensure vis dir exists
    model.train()
    salEvalVal = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []
    total_batches = len(train_loader)

    for iter, batched_inputs in enumerate(train_loader):
        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        start_time = time.time()
        lr = adjust_learning_rate(args, optimizer, epoch, iter + cur_iter, max_batches, lr_factor=lr_factor)

        if args.onGPU and torch.cuda.is_available():
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        pre_img_var  = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var   = torch.autograd.Variable(target).float()

        output = model(pre_img_var, post_img_var)
        loss = BCEDiceLoss(output, target_var)
        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        time_taken = time.time() - start_time
        res_time = (max_batches * args.max_epochs - (iter + cur_iter)) * time_taken / 3600

        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(pred, 0, dim=0)

        with torch.no_grad():
            f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=target_var.cpu().numpy())

        if iter % 5 == 0:
            print('\riteration: [%d/%d] f1: %.3f lr: %.7f loss: %.3f time:%.3f h' %
                  (iter + cur_iter, max_batches * args.max_epochs, f1, lr, loss.item(), res_time), end='')

        if np.mod(iter, 200) == 1:
            vis_input  = utils.make_numpy_grid(utils.de_norm(pre_img_var[0:8]))
            vis_input2 = utils.make_numpy_grid(utils.de_norm(post_img_var[0:8]))
            vis_pred   = utils.make_numpy_grid(pred[0:8])
            vis_gt     = utils.make_numpy_grid(target_var[0:8])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(args.vis_dir, f'train_{epoch}_{iter}.jpg')
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            plt.imsave(file_name, vis)

    average_epoch_loss_train = sum(epoch_loss) / max(1, len(epoch_loss))
    scores = salEvalVal.get_scores()
    return average_epoch_loss_train, scores, lr


def adjust_learning_rate(args, optimizer, epoch, iter, max_batches, lr_factor=1):
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step_loss))
    elif args.lr_mode == 'poly':
        cur_iter = iter
        max_iter = max(1, max_batches * args.max_epochs)
        lr = args.lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError(f'Unknown lr mode {args.lr_mode}')
    if epoch == 0 and iter < 200:
        lr = args.lr * 0.9 * (iter + 1) / 200 + 0.1 * args.lr  # warm-up
    lr *= lr_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def trainValidateSegmentation(args):
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    model = BaseNet(3, 1)

    # ---- Safe run folder name (avoid slashes in savedir) ----
    dataset_name = os.path.basename(os.path.normpath(args.file_root))
    args.savedir = os.path.join(args.savedir, f'{dataset_name}_iter_{args.max_steps}_lr_{args.lr}')
    args.vis_dir = os.path.join(args.savedir, 'Vis')

    # ---- Resolve dataset root ----
    if args.file_root in ('LEVIR','BCDD','SYSU','CDD','quick_start'):
        aliases = {
            'LEVIR':'H:\\penghaifeng\\LEVIR-CD',
            'BCDD':'H:\\penghaifeng\\BCDD',
            'SYSU':'H:\\penghaifeng\\SYSU-CD',
            'CDD':'/home/guan/Documents/Datasets/ChangeDetection/CDD',
            'quick_start':'./samples'
        }
        args.file_root = aliases[args.file_root]
    else:
        if not os.path.isdir(args.file_root):
            raise TypeError(f'{args.file_root} has not defined')

    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    if args.onGPU and torch.cuda.is_available():
        model = model.cuda()

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters (excluding idr): ' + str(total_params))

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std  = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    # transforms
    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.RandomCropResize(int(7. / 224. * args.inWidth)),
        myTransforms.RandomFlip(),
        myTransforms.RandomExchange(),
        myTransforms.ToTensor()
    ])

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    train_data = myDataLoader.Dataset("train", file_root=args.file_root, transform=trainDataset_main)
    trainLoader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False, drop_last=True
    )

    val_data = myDataLoader.Dataset("val", file_root=args.file_root, transform=valDataset)
    valLoader = torch.utils.data.DataLoader(
        val_data, shuffle=False, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=False
    )

    # Optional test split
    has_test = True
    try:
        test_data = myDataLoader.Dataset("test", file_root=args.file_root, transform=valDataset)
        testLoader = torch.utils.data.DataLoader(
            test_data, shuffle=False, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=False
        )
    except Exception as e:
        print("[Info] Test split unavailable or unlabeled — skipping test phase.\nReason:", e)
        has_test = False
        testLoader = None

    max_batches = max(1, len(trainLoader))
    print('For each epoch, we have {} batches'.format(max_batches))

    if args.onGPU and torch.cuda.is_available():
        cudnn.benchmark = True

    # ensure at least 1 epoch
    args.max_epochs = max(1, int(np.ceil(args.max_steps / max_batches)))
    start_epoch = 0
    cur_iter = 0
    max_F1_val = -1.0  # so first val can become best

    # resume (only if requested truthy AND file exists)
    if isinstance(args.resume, str) and args.resume.lower() in ('true','1','yes'):
        resume_path = os.path.join(args.savedir, 'checkpoint.pth.tar')
        if os.path.isfile(resume_path):
            print(f"=> loading checkpoint '{resume_path}'")
            checkpoint = torch.load(resume_path, map_location='cuda' if args.onGPU and torch.cuda.is_available() else 'cpu')
            start_epoch = checkpoint.get('epoch', 0)
            cur_iter = start_epoch * len(trainLoader)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded checkpoint '{resume_path}' (epoch {checkpoint.get('epoch', '?')})")
        else:
            print(f"=> no checkpoint found at '{resume_path}'")

    logFileLoc = os.path.join(args.savedir, args.logFile)
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_params)))
        logger.write("\n%s\t%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Kappa (val)', 'IoU (val)', 'F1 (val)', 'R (val)', 'P (val)'))
    logger.flush()

    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)

    model_file_name = os.path.join(args.savedir, 'best_model.pth')  # defined early

    for epoch in range(start_epoch, args.max_epochs):
        lossTr, score_tr, lr = train(args, trainLoader, model, optimizer, epoch, max_batches, cur_iter)
        cur_iter += len(trainLoader)
        torch.cuda.empty_cache()

        # ✅ Do validation even on epoch 0 (so short runs still save a model)
        lossVal, score_val = val(args, valLoader, model, epoch)
        torch.cuda.empty_cache()

        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" %
                     (epoch, score_val['Kappa'], score_val['IoU'], score_val['F1'],
                      score_val['recall'], score_val['precision']))
        logger.flush()

        # always save a rolling checkpoint
        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lossTr': lossTr,
            'lossVal': lossVal,
            'F_Tr': score_tr['F1'],
            'F_val': score_val['F1'],
            'lr': lr
        }, os.path.join(args.savedir, 'checkpoint.pth.tar'))

        # update best
        if score_val['F1'] >= max_F1_val:
            max_F1_val = score_val['F1']
            torch.save(model.state_dict(), model_file_name)

        # also save a last snapshot every epoch (useful for tiny runs)
        torch.save(model.state_dict(), os.path.join(args.savedir, 'last_model.pth'))

        print("Epoch " + str(epoch) + ': Details')
        print("\nEpoch No. %d:\tTrain Loss = %.4f\tVal Loss = %.4f\t F1(tr) = %.4f\t F1(val) = %.4f" %
              (epoch, lossTr, lossVal, score_tr['F1'], score_val['F1']))
        torch.cuda.empty_cache()

    # Load best if exists, else keep current params
    if os.path.isfile(model_file_name):
        state_dict = torch.load(model_file_name, map_location=('cuda' if args.onGPU and torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(state_dict)
    else:
        print('[WARN] No best_model.pth available; using last epoch weights.')

    # optional test
    if has_test and testLoader is not None:
        loss_test, score_test = val(args, testLoader, model, 0)
        print("\nTest :\t Kappa (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f" %
              (score_test['Kappa'], score_test['IoU'], score_test['F1'], score_test['recall'], score_test['precision']))
        logger.write("\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" %
                     ('Test', score_test['Kappa'], score_test['IoU'], score_test['F1'],
                      score_test['recall'], score_test['precision']))
        logger.flush()
    else:
        print("\n[Info] Skipped test evaluation (no test split found).")

    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default="LEVIR", help='Data directory | LEVIR | BCDD | SYSU ')
    parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
    parser.add_argument('--max_steps', type=int, default=40000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy, step or poly')
    parser.add_argument('--savedir', default='H:\\penghaifeng\\A2Net-main2\\results', help='Directory to save the results')
    parser.add_argument('--resume', default=True, help='Resume from checkpoint in savedir (True/False or path)')
    parser.add_argument('--logFile', default='trainValLog.txt', help='File that stores the training and validation logs')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'), help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weight', default='', type=str, help='pretrained weight, can be a non-strict copy')
    parser.add_argument('--ms', type=int, default=0, help='apply multi-scale training, default False')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    trainValidateSegmentation(args)
