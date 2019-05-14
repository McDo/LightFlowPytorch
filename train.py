import argparse
import os
import time

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from utils import flow_transforms
import models
import datasets
from loss import multiscaleEPE, realEPE, L1Loss
import datetime
from tensorboardX import SummaryWriter
from utils.util import AverageMeter, save_checkpoint, adjust_learning_rate
from utils.flowlib import flow2rgb


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))
dataset_names = sorted(name for name in datasets.__all__)


parser = argparse.ArgumentParser(description='Light FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR', type=str, default='/path/to/dataset',
                    help='path to dataset')
# parser.add_argument('--dataset', metavar='DATASET', default='mpi_sintel_final', choices=dataset_names,
parser.add_argument('--dataset', metavar='DATASET', default='flying_chairs', choices=dataset_names,
                    help='dataset type : ' + ' | '.join(dataset_names))
group = parser.add_mutually_exclusive_group()
group.add_argument('-s', '--split-file', default=None, type=str,
                   help='test-val split file')
group.add_argument('--split-value', default=0.97, type=float,
                   help='test-val split proportion between 0 (only test) and 1 (only train), '
                        'will be overwritten if a split file is set')
parser.add_argument('--arch', '-a', metavar='ARCH', default='flownetl', choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' +
                    ' | '.join(model_names))
parser.add_argument('--solver', default='adam', choices=['adam', 'sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=-1, type=int, metavar='N',
                    help='number of total epochs to run (will match 70k iterations in total if set to -1)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-5, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--sparse', action='store_true',
                    help='look for NaNs in target flow when computing EPE, avoid if flow is garantied to be dense,'
                    'automatically seleted when choosing a KITTIdataset')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--no-date', action='store_true',
                    help='don\'t append date timestamp to folder' )
parser.add_argument('--div-flow', default=20,
                    help='value by which flow will be divided. '
                         'Original value is 20 but 1 with batchNorm gives good results')

best_EPE = -1
n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    global args, best_EPE
    args = parser.parse_args()
    save_path = '{},{},b{},lr{}'.format(
        args.arch,
        args.solver,
        args.batch_size,
        args.lr
    )
    if not args.no_date:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = os.path.join(timestamp, save_path)
    save_path = os.path.join(args.dataset, save_path)
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(os.path.join(save_path, 'test', str(i))))

    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])
    target_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0], std=[args.div_flow, args.div_flow])
    ])

    if 'KITTI' in args.dataset:
        args.sparse = True
    if args.sparse:
        co_transform = flow_transforms.Compose([
            flow_transforms.RandomCrop((320, 448)),
            flow_transforms.RandomVerticalFlip(),
            flow_transforms.RandomHorizontalFlip()
        ])
    else:
        co_transform = flow_transforms.Compose([
            flow_transforms.RandomTranslate(10),
            flow_transforms.RandomRotate(10, 5),
            flow_transforms.RandomCrop((320, 448)),
            flow_transforms.RandomVerticalFlip(),
            flow_transforms.RandomHorizontalFlip()
        ])

    print("=> fetching img pairs in '{}'".format(args.data))
    train_set, test_set = datasets.__dict__[args.dataset](
        args.data,
        transform=input_transform,
        target_transform=target_transform,
        co_transform=co_transform,
        split=args.split_file if args.split_file else args.split_value
    )
    print('{} samples found, {} train samples and {} test samples '.format(len(test_set) + len(train_set),
                                                                           len(train_set),
                                                                           len(test_set)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False
    )

    # create model
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        args.arch = network_data['arch']
        print("=> using pre-trained model '{}'".format(args.pretrained))
    else:
        network_data = None
        print("=> creating model '{}'".format(args.arch))

    model = models.__dict__[args.arch](network_data)
    bias_params = model.bias_parameters()
    weight_params = model.weight_parameters()
    parallel = False

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            parallel = True
        model = model.to(device)
        cudnn.benchmark = True

        if parallel is True:
            bias_params = model.module.bias_parameters()
            weight_params = model.module.weight_parameters()

    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    param_groups = [{'params': bias_params, 'weight_decay': args.bias_decay},
                    {'params': weight_params, 'weight_decay': args.weight_decay}]
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr,
                                     betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr,
                                    momentum=args.momentum)

    if args.evaluate:
        best_EPE = validate(val_loader, model, 0, output_writers)
        return

    epoch_size = len(train_loader)
    # device_num = torch.cuda.device_count() if torch.cuda.is_available() else 1
    # epochs = 70000 // device_num // epoch_size + 1 if args.epochs == -1 else args.epochs
    epochs = 70000 // epoch_size + 1 if args.epochs == -1 else args.epochs

    for epoch in range(args.start_epoch, epochs):
        adjust_learning_rate(args, optimizer, epoch, epoch_size)

        # train for one epoch
        train_loss, train_EPE = train(train_loader, model, optimizer, epoch, epoch_size, train_writer)
        train_writer.add_scalar('mean EPE', train_EPE, epoch)

        # evaluate on validation set
        with torch.no_grad():
            EPE = validate(val_loader, model, epoch, output_writers)
        test_writer.add_scalar('mean EPE', EPE, epoch)

        if best_EPE < 0:
            best_EPE = EPE

        is_best = EPE < best_EPE
        best_EPE = min(EPE, best_EPE)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'best_EPE': best_EPE,
            'div_flow': args.div_flow
        }, is_best, save_path)


def train(train_loader, model, optimizer, epoch, epoch_size, train_writer):
    global n_iter, args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow_finest_EPEs = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.to(device)
        input = torch.cat(input, 1).to(device)

        # compute output
        output = model(input)

        if args.sparse:
            # Since Target pooling is not very precise when sparse,
            # take the highest resolution prediction and upsample it instead of downsampling target
            h, w = target.size()[-2:]
            output = [F.interpolate(output[0], (h, w)), *output[1:]]

        loss = L1Loss(output, target, sparse=args.sparse)
        flow_finest_EPE = args.div_flow * realEPE(output, target, sparse=args.sparse)

        # record loss and EPE
        losses.update(loss.item(), target.size(0))
        train_writer.add_scalar('train_loss', loss.item(), n_iter)
        flow_finest_EPEs.update(flow_finest_EPE.item(), target.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t Time(s) {3}\t Data Time(s) {4}\t Loss {5}\t EPE {6}'
                  .format(epoch, i, epoch_size, batch_time,
                          data_time, losses, flow_finest_EPEs))
        n_iter += 1
        if i >= epoch_size:
            break

    return losses.avg, flow_finest_EPEs.avg


def validate(val_loader, model, epoch, output_writers):
    global args

    batch_time = AverageMeter()
    flow_finest_EPEs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = torch.cat(input,1).to(device)

        # compute output
        output = model(input)
        flow_finest_EPE = args.div_flow * realEPE(output, target, sparse=args.sparse)
        # record EPE
        flow_finest_EPEs.update(flow_finest_EPE.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i < len(output_writers):  # log first output of first batches
            if epoch == 0:
                mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=input.dtype).view(3, 1, 1)
                output_writers[i].add_image('GroundTruth', flow2rgb(args.div_flow * target[0], max_value=10), 0)
                output_writers[i].add_image('Inputs', (input[0,:3].cpu() + mean_values).clamp(0, 1), 0)
                output_writers[i].add_image('Inputs', (input[0,3:].cpu() + mean_values).clamp(0, 1), 1)
            output_writers[i].add_image('FlowNet Outputs', flow2rgb(args.div_flow * output[0], max_value=10), epoch)

        print('Test: [{0}/{1}]\t Time(s) {2}\t EPE {3}'
              .format(i, len(val_loader), batch_time, flow_finest_EPEs))

    print(' * EPE {:.3f}'.format(flow_finest_EPEs.avg))

    return flow_finest_EPEs.avg


if __name__ == '__main__':
    main()
