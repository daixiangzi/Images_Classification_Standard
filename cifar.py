'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
python3 cifar.py
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

from tensorboardX import SummaryWriter
from config import Config
from MyDataset import MyDataset
from utils.radam import RAdam,AdamW,Lookahead
from utils.mix_up import mixup_data,mixup_criterion
from Cutout import Cutout
from utils.loss import Label_smoothing
opt = Config()
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
use_cuda = torch.cuda.is_available()
state = {'lr':opt.lr}
writer = SummaryWriter(log_dir=opt.logs)
# Random seed
if opt.seed is None:
    opt.seed = random.randint(1, 10000)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
if use_cuda:
    torch.cuda.manual_seed_all(opt.seed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    if not os.path.isdir(opt.save):
        mkdir_p(opt.save)
    if not os.path.isdir(opt.logs):
        mkdir_p(opt.logs)



    # Data
    if opt.cutout:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Cutout(opt.cutout_n,opt.cutout_len),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = MyDataset(opt.train_data,transform=transform_train)
    #trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=opt.train_batch, shuffle=True, num_workers=opt.workers)

    #testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testset = MyDataset(opt.test_data,transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=opt.test_batch, shuffle=False, num_workers=opt.workers)

    # Model
    model = models.__dict__[opt.arch](
                    num_classes=opt.num_classes,
                    depth=opt.depth,
                    block_name='BasicBlock'#BasicBlock, Bottleneck,
                )
    if opt.init=='kaiming':
        model.apply(weights_init)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    if opt.optim=="SGD":
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optim=="Adam":
        optimizer = optim.Adam(model.parameters(),lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=opt.weight_decay)
    elif opt.optim=='RAdam':
        optimizer = RAdam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
    elif opt.optim=='AdamW':
        optimizer = AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay, warmup=opt.decay_epoch)
    if opt.lookahead:
        optimizer = Lookahead(optimizer, la_steps=opt.la_steps, la_alpha=opt.la_alpha)
    # Resume
    title = 'cifar-10-' + opt.arch
    if opt.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(opt.resume), 'Error: no checkpoint directory found!'
        opt.save = os.path.dirname(opt.resume)
        checkpoint = torch.load(opt.resume)
        best_acc = checkpoint['best_acc']
        opt.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(opt.save, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(opt.save, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if opt.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, opt.start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) 
    if opt.label_smooth:
        criterion = Label_smoothing(opt.num_classes,opt.esp)
    else:
        criterion = nn.CrossEntropyLoss()
    for epoch in range(opt.start_epoch, opt.epochs):
        adjust_learning_rate(optimizer, epoch)
        #if epoch%20==0:
         #   scheduler.step()
        if opt.warmming_up and epoch <= opt.decay_epoch:
            warmming_up(optimizer,epoch)

        #print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
        if (epoch+1)%opt.fre_print==0:
            print("%d epoch,train_loss:%f,test_loss:%f,train_acc:%f,test_acc:%f,best_acc:%f,lr:%f"%(epoch,train_loss,test_loss,train_acc,test_acc,best_acc,state['lr']))
        #viso
        writer.add_scalar('train/acc',train_acc,epoch)
        writer.add_scalar('train/loss',train_loss,epoch)

        writer.add_scalar('test/acc',test_acc,epoch)
        writer.add_scalar('test/loss',test_loss,epoch)

        writer.add_scalar('learn_rate',state['lr'])
        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=opt.save)

    logger.close()
    logger.plot()
    savefig(os.path.join(opt.save, 'log.eps'))
    writer.close()
    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    #bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        if opt.mix_up:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, opt.alpha)
            loss_func = mixup_criterion(targets_a, targets_b, lam)
            outputs = model(inputs)
            loss = loss_func(criterion, outputs)
        # compute output
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
       
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        """
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    """
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    #bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        """
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    """
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in opt.schedule:
        state['lr'] *= opt.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
def warmming_up(optimizer,epoch):
    state['lr'] = opt.lr*(epoch+1)/opt.decay_epoch
    for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv')!=-1 or classname.find('Linear') != -1:
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find("BatchNorm")!=-1:
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
if __name__ == '__main__':
    main()
