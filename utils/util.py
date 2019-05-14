import os
import numpy as np
import shutil
import torch


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def adjust_learning_rate(args, optimizer, epoch, epoch_size):
    """
    Sets the learning rate
    Adapted from PyTorch Imagenet example:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    first10k = every10k = 10000 // epoch_size + 1
    second20k = first10k + 20000 // epoch_size + 1

    if epoch <= first10k:
        lr = args.lr
    elif first10k < epoch <= second20k:
        lr = args.lr * 0.5
    elif second20k < epoch:
        lr = args.lr * 0.5 / (2 ** ((epoch - second20k) // every10k + 1))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.lr = 0.001

    epoch_size = 174
    args = Args()

    # warmup
    first10k = every10k = 10000 // epoch_size + 1
    second20k = first10k + 20000 // epoch_size + 1
    print(f"first10k: {first10k}")
    print(f"second20k: {second20k}")
    print(f"every10k: {every10k}\n")

    for epoch in range(0, 403):
        if epoch <= first10k:
            lr = args.lr
            print(f"epoch: {epoch} <= first10k: {first10k}, lr: {lr}")
        elif first10k < epoch <= second20k:
            lr = args.lr * 0.5
            print(f"first10k: {first10k} < epoch: {epoch} <= second20k: {second20k}, lr: {lr}")
        elif second20k < epoch:
            lr = args.lr * 0.5 / (2 ** ((epoch - second20k) // every10k + 1))
            print(f"second10k: {second20k} < epoch: {epoch}, lr: {lr}")
