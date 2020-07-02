import argparse
import os
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

# import torchvision
# import torchvision.transforms as transforms

from models import *
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-a', '--arch', metavar='ARCH', default='res20')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 10)')
# parser.add_argument('--init', help='initialize form pre-trained floating point model', type=str, default='./result/res20_3bit/model_best.pth.tar')
parser.add_argument('--init', help='initialize form pre-trained floating point model', type=str, default='')
parser.add_argument('-id', '--device', default='0', type=str, help='gpu device')
parser.add_argument('--bit', default=3, type=int, help='the bit-width of the quantized network')

best_prec = 0
args = parser.parse_args()

def main():

    global args, best_prec
    # use_gpu = torch.cuda.is_available()
    use_gpu = True
    print(args.device)
    print('=> Building model...')
    model=None
    if use_gpu:
        float = True if args.bit == 32 else False
        if args.arch == 'res20':
            model = resnet20_cifar(float=float)
        elif args.arch == 'res56':
            model = resnet56_cifar(float=float)
        else:
            print('Architecture not support!')
            return
        if not float:
            for m in model.modules():
                if isinstance(m, QuantConv2d):
                    init_alpha = m.weight.abs().max()*0.7
                    m.weight_quant = weight_quantize_fn(w_bit=args.bit, init_alpha=init_alpha)
                    m.act_grid = build_power_value(args.bit)
                    m.act_alq = act_quantization(args.bit, m.act_grid)

        model = nn.DataParallel(model)
        criterion = nn.CrossEntropyLoss()
        model_params = []
        for name, params in model.module.named_parameters():
            if 'act_alpha' in name:
                model_params += [{'params': [params], 'lr': 1e-1, 'weight_decay': 1e-4}]
            elif 'wgt_alpha' in name:
                model_params += [{'params': [params], 'lr': 2e-2, 'weight_decay': 1e-4}]
            else:
                model_params += [{'params': [params]}]
        optimizer = torch.optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        cudnn.benchmark = True
    else:
        print('Cuda is not available!')
        return

    if not os.path.exists('result'):
        os.makedirs('result')
    fdir = 'result/'+str(args.arch)+'_'+str(args.bit)+'bit'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    if args.init:
        if os.path.isfile(args.init):
            print("=> loading pre-trained model")
            checkpoint = torch.load(args.init, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'],strict=False)
        else:
            print('No pre-trained model found !')
            exit()

    print('=> loading cifar10 data...')
    # normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    # train_dataset = torchvision.datasets.CIFAR10(
    #     root='/Users/hlbai/Research/datasets/',
    #     train=True,
    #     download=False,
    #     transform=transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # test_dataset = torchvision.datasets.CIFAR10(
    #     root='/Users/hlbai/Research/datasets/',
    #     train=False,
    #     download=False,
    #     transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    # testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    # if args.evaluate:
    #     validate(testloader, model, criterion)
    #     model.module.show_params()
    #     return
    # writer = SummaryWriter(comment=fdir.replace('result/', ''))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        # model.module.record_weight(writer, epoch)
        if epoch%10 == 1:
            model.module.show_params()
        # model.module.record_clip(writer, epoch)

        # add act hooks for plot
        # act_collectors = {}
        # act_collectors, hooks = prepare_hooks(model, act_collectors)

        train(None, model, criterion, optimizer, epoch)

        # plot distributions
        # plot_wgt_distribution(model)
        # plot_act_distribution(model, act_collectors)
        # remove_hooks(hooks)

        # evaluate on test set
        prec = validate(None, model, criterion)
        # writer.add_scalar('test_acc', prec, epoch)

        # remember best precision and save checkpoint
        is_best = prec > best_prec
        best_prec = max(prec,best_prec)
        print('best acc: {:1f}'.format(best_prec))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, fdir)


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


def train(trainloader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    # for i, (input, target) in enumerate(trainloader):
    for i in range(10):
        # measure data loading time
        data_time.update(time.time() - end)

        input = torch.randn(8, 3, 32, 32)
        target = torch.randint(0, 9, size=(8,), dtype=torch.int64)
        input, target = input, target

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # if i % 2 == 0:
        #     model.module.show_params()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   epoch, i, -1, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i in range(5):
            input = torch.randn(8, 3, 32, 32)
            target = torch.randint(0, 9, size=(8,), dtype=torch.int64)
            input, target = input, target

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   i, -1, batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec {top1.avg:.3f}% '.format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    adjust_list = [150, 225]
    if epoch in adjust_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def plot_wgt_distribution(model, epoch=0):
    print('Ploting weight distributions...')
    layer_idx = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            layer_idx += 1
            kernel = m.weight.view(-1).detach().numpy()
            mu, std = norm.fit(kernel)
            plt.hist(kernel, bins=25, density=True, alpha=0.6, color='b')

            xlim = 1.5*np.abs(kernel).max()
            x = np.linspace(-xlim, xlim, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2)
            title = "Layer%d: mu = %.2f,  std = %.2f" % (layer_idx, mu, std)
            plt.title(title)
            plt.savefig('./plots/layer_%d_wgt_distribution.png'%layer_idx)
            plt.close()


def prepare_hooks(model, act_collectors):

    def add_act_hook(module, input_, output_):
        if act_collectors.has_key(id(module)):
            act_collectors[id(module)].append(input_[0].detach().numpy())
        else:
            act_collectors[id(module)] = [input_[0].detach().numpy()]

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hook = m.register_forward_hook(add_act_hook)
            hooks.append(hook)
    return act_collectors, hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def plot_act_distribution(model, act_collectors, epoch=0):
    print('Ploting act distributions...')
    layer_idx = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            layer_idx += 1
            act_data = act_collectors[id(m)]
            assert len(act_data) != 0, "empty act collectors"
            act_data = np.concatenate(act_data, axis=0)
            act_data = act_data.reshape(-1)
            plt.hist(act_data, bins=25, density=True, alpha=0.6, color='b')
            title = "Layer%d activation" % (layer_idx)
            plt.title(title)
            plt.savefig('./plots/layer_%d_act_distribution.png'%layer_idx)
            plt.close()


if __name__=='__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main()