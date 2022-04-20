import os
import sys
import argparse
from torch.utils.data import DataLoader
from data_processing.generate_dataset import DatasetCategory
from data_processing.hierarchical_dataset import DeepFashionHierarchihcalDataset, HierarchicalBatchSampler
from torch.optim import lr_scheduler
from util.util import adjust_learning_rate, warmup_learning_rate, TwoCropTransform
from losses.losses import HMLC
from network import resnet_modified
from network.resnet_modified import LinearClassifier
import tensorboard_logger as tb_logger
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.distributed as dist
import time
import shutil
import math
import builtins

def parse_option():
    parser = argparse.ArgumentParser(description='Training/finetuning on Deep Fashion Dataset')
    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset, the superset of train/val')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--train-listfile', default='', type=str,
                        help='training file with annotation')
    parser.add_argument('--val-listfile', default='', type=str,
                        help='validation file with annotation')
    parser.add_argument('--class-map-file', default='', type=str,
                        help='class mapping between str and int')
    parser.add_argument('--class-seen-file', default='', type=str,
                        help='seen classes text file. Used for seen/unseen split experiments.')
    parser.add_argument('--class-unseen-file', default='', type=str,
                        help='unseen classes text file. Used for seen/unseen split experiments.')
    parser.add_argument('--repeating-product-file', default='', type=str,
                        help='repeating product ids file')
    parser.add_argument('--mode', default='train', type=str,
                        help='Train or val')
    parser.add_argument('--input-size', default=224, type=int,
                        help='input size')
    parser.add_argument('--scale-size', default=256, type=int,
                        help='scale size in validation')
    parser.add_argument('--crop-size', default=224, type=int,
                        help='crop size')
    parser.add_argument('--num-classes', type=int,
                        help='number of classes')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        metavar='N', help='mini-batch size (default: 512)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                        help='use pre-trained model')
    parser.add_argument('--feature-extract', action='store_false',
                        help='When flase, finetune the whole model; else only update the reshaped layer para')
    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,60,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    #other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--loss', type=str, default='hmce',
                        help='loss type', choices=['hmc', 'hce', 'hmce'])
    parser.add_argument('--tag', type=str, default='',
                        help='tag for model name')
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    # warm-up for large-batch training,
    if args.batch_size >= 256:
        args.warm = True
    if args.warm:
        args.model_name = '{}_warm'.format(args.model)
        args.warmup_from = 0.01
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate
    return args

best_prec1 = 0

def main():
    global args, best_prec1
    args = parse_option()

    args.save_folder = './model'
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    args.tb_folder = './tensorboard'
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    args.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_loss_{}_trial_{}'.\
        format('hmlc', 'dataset', args.model, args.learning_rate,
               args.lr_decay_rate, args.batch_size, args.loss, 5)
    if args.tag:
        args.model_name = args.model_name + '_tag_' + args.tag
    args.tb_folder = os.path.join(args.tb_folder, args.model_name)
    args.save_folder = os.path.join(args.save_folder, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    # distributed training
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        print("Adopting distributed multi processing training")
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    print("GPU in main worker is {}".format(gpu))
    args.gpu = gpu
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            print("In the process of multi processing with rank as {}".format(args.rank))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.model))
    model, criterion = set_model(ngpus_per_node, args)
    args.classifier = LinearClassifier(name=args.model, num_classes=args.num_classes).cuda(args.gpu)
    set_parameter_requires_grad(model, args.feature_extract)
    optimizer = setup_optimizer(model, args.learning_rate,
                                   args.momentum, args.weight_decay,
                                   args.feature_extract)
    cudnn.benchmark = True

    dataloaders_dict, sampler = load_deep_fashion_hierarchical(args.data, args.train_listfile,
                                 args.val_listfile, args.class_map_file, args.repeating_product_file,
                                 args)

    train_sampler, val_sampler = sampler['train'], sampler['val']
    for epoch in range(1, args.epochs + 1):
        print('Epoch {}/{}'.format(epoch, args.epochs + 1))
        print('-' * 10)
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        train(dataloaders_dict, model, criterion, optimizer, epoch, args, logger)
        output_file = args.save_folder + '/checkpoint_{:04d}.pth.tar'.format(epoch)

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False,
                filename=output_file)

def set_model(ngpus_per_node, args):
    model = resnet_modified.MyResNet(name='resnet50')
    criterion = HMLC(temperature=args.temp, loss_type=args.loss, layer_penalty=torch.exp)

    # This part is to load a pretrained model
    ckpt = torch.load(args.ckpt, map_location='cpu')
    # state_dict = ckpt['state_dict']
    state_dict = ckpt['model']
    model_dict = model.state_dict()
    new_state_dict = {}
    # for k, v in state_dict.items():
    #     if not k.startswith('module.encoder_q.fc'):
    #         k = k.replace('module.encoder_q', 'encoder')
    #         new_state_dict[k] = v
    for k, v in state_dict.items():
        if not k.startswith('module.head'):
            k = k.replace('module.encoder', 'encoder')
            new_state_dict[k] = v
    state_dict = new_state_dict
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        print("GPU setting", args.gpu)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            print("Updated batch size is {}".format(args.batch_size))
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            # There is memory issue in data loader
            # args.workers = 0
            model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu],find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            print('Loading state dict from ckpt')
            model.load_state_dict(state_dict)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    criterion = criterion.cuda(args.gpu)

    return model, criterion

def train(dataloaders, model, criterion, optimizer, epoch, args, logger):
    """
    one epoch training
    """

    classifier = args.classifier
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    end = time.time()

    # Each epoch has a training and/or validation phase
    for phase in ['train']:
        if phase == 'train':
            progress = ProgressMeter(len(dataloaders['train']),
                [batch_time, data_time, losses, top1, top5],
                prefix="Epoch: [{}]".format(epoch))
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode
        classifier.eval()

        # Iterate over data.
        for idx, (images, labels) in enumerate(dataloaders[phase]):
            data_time.update(time.time() - end)
            labels = labels.squeeze()
            images = torch.cat([images[0].squeeze(), images[1].squeeze()], dim=0)
            images = images.cuda(non_blocking=True)
            labels = labels.squeeze().cuda(non_blocking=True)
            bsz = labels.shape[0] #batch size
            if phase == 'train':
                warmup_learning_rate(args, epoch, idx, len(dataloaders[phase]), optimizer)

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                # Get model outputs and calculate loss
                features = model(images)
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = criterion(features, labels)
                losses.update(loss.item(), bsz)

                # backward + optimize only if in training phase
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            sys.stdout.flush()
            if idx % args.print_freq == 0:
                progress.display(idx)
        logger.log_value('loss', losses.avg, epoch)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_data(root_dir, train_listfile, val_listfile, class_map_file,
              class_seen_file, class_unseen_file, input_size, scale_size, crop_size, batch_size, distributed, workers):
    # Data augmentation and normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # Create training and validation datasets
    train_dataset = DatasetCategory(root_dir, 'train', train_listfile, val_listfile, '',
                                    class_map_file, class_seen_file,
                                    class_unseen_file, TwoCropTransform(data_transforms['train']))
    val_dataset = DatasetCategory(root_dir, 'val', train_listfile, val_listfile, '',
                                  class_map_file, class_seen_file,
                                  class_unseen_file, TwoCropTransform(data_transforms['val']))
    image_datasets = {'train': train_dataset,
                      'val': val_dataset}
    print("Initializing Datasets and Dataloaders...")

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None
    sampler = {'train': train_sampler,
                'val': val_sampler}

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                       shuffle=(train_sampler is None), num_workers=workers,
                                       pin_memory=True, sampler=sampler[x], drop_last=True)
        for x in ['train', 'val']}
    return dataloaders_dict, sampler


def load_deep_fashion_hierarchical(root_dir, train_list_file, val_list_file, class_map_file, repeating_product_file, opt):
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.input_size, scale=(0.8, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4)
            ], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [
                                                            0.229, 0.224, 0.225]),
                                       ])
    train_dataset = DeepFashionHierarchihcalDataset(os.path.join(root_dir, train_list_file),
                                                    os.path.join(root_dir, class_map_file),
                                                    os.path.join(root_dir, repeating_product_file),
                                                    transform=TwoCropTransform(train_transform))
    
    val_dataset = DeepFashionHierarchihcalDataset(os.path.join(root_dir, val_list_file),
                                                  os.path.join(
                                                      root_dir, class_map_file),
                                                  os.path.join(
                                                      root_dir, repeating_product_file),
                                                  transform=TwoCropTransform(val_transform))
    print('LENGTH TRAIN', len(train_dataset))
    image_datasets = {'train': train_dataset,
                      'val': val_dataset}
    train_sampler = HierarchicalBatchSampler(batch_size=opt.batch_size,
                                       drop_last=False,
                                       dataset=train_dataset)
    val_sampler = HierarchicalBatchSampler(batch_size=opt.batch_size,
                                           drop_last=False,
                                           dataset=val_dataset)
    sampler = {'train': train_sampler,
               'val': val_sampler}
    print(opt.workers, "workers")
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], sampler=sampler[x],
                                       num_workers=opt.workers, batch_size=1,
                                       pin_memory=True)
        for x in ['train', 'val']}
    return dataloaders_dict, sampler


def setup_optimizer(model_ft, lr, momentum, weight_decay, feature_extract):
    # Send the model to GPU
    # model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer_ft

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        # Select which params to finetune
        # for param in model.parameters():
        #     param.requires_grad = True
        for name, param in model.module.named_parameters():
            if name.startswith('encoder.layer4'):
                param.requires_grad = True
            elif name.startswith('encoder.layer3'):
                param.requires_grad = True
            elif name.startswith('head'):
                param.requires_grad = True
            else:
                param.requires_grad = False

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



if __name__ == '__main__':
    main()
