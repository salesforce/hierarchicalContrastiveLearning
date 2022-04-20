import os
import math
import argparse
from data_processing.generate_dataset import DatasetCategory
from data_processing.hierarchical_dataset import DeepFashionHierarchihcalDatasetEval
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from network.resnet_modified import LinearClassifier
from util.util import adjust_learning_rate, warmup_learning_rate, set_optimizer
from network import resnet_modified
import time
import sys
import shutil
import tensorboard_logger as tb_logger

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset, the superset of train/val')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--train-listfile', default='', type=str,
                        help='train file with annotation')
    parser.add_argument('--val-listfile', default='', type=str,
                        help='validation file with annotation')
    parser.add_argument('--test-listfile', default='', type=str,
                        help='test file with annotation')
    parser.add_argument('--class-seen-file', default='', type=str,
                        help='seen classes text file')
    parser.add_argument('--class-unseen-file', default='', type=str,
                        help='unseen classes text file')
    parser.add_argument('--repeating-product-file', default='', type=str,
                        help='repeating product ids file')
    parser.add_argument('--mode', default='val', type=str,
                        help='test or val')
    parser.add_argument('--class-map-file', default='', type=str,
                        help='class mapping between str and int')
    parser.add_argument('--input-size', default=224, type=int,
                        help='input size')
    parser.add_argument('--scale-size', default=256, type=int,
                        help='scale size in validation')
    parser.add_argument('--crop-size', default=224, type=int,
                        help='crop size')
    parser.add_argument('--num-classes', type=int,
                        help='number of classes')
    parser.add_argument('--epochs', type=int, default=100,
                            help='number of training epochs')

    parser.add_argument('--ckpt', type=str,
                        help='the pth file to load')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                        help='use pre-trained model')
    parser.add_argument('--feature-extract', action='store_false',
                        help='When flase, finetune the whole model; else only update the reshaped layer para')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,60,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--tag', type=str, default='',
                        help='tag for model name')
                        
    args = parser.parse_args()
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    return args


def main():
    global args
    args = parse_option()
    best_acc = 0
    args.save_folder = './model_linear/'
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    args.tb_folder = './tensorboard'
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    args.model_name = '{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
        format('hmlc', 'dataset', 'resnet50', os.path.split(args.ckpt)[1], args.learning_rate,
               args.lr_decay_rate, args.batch_size, 5)
    if args.tag:
        args.model_name = args.model_name + '_tag_' + args.tag

    if args.warm:
        args.model_name = '{}_warm'.format(args.model_name)
        args.warmup_from = 0.01
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate
    args.tb_folder = os.path.join(args.tb_folder, args.model_name)
    args.save_folder = os.path.join(args.save_folder, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # build model and criterion
    model, classifier, criterion = set_model(args)
    cudnn.benchmark = True
    dataloaders_dict,_ = load_deep_fashion_hierarchical(args.data, args.train_listfile,
                                                               args.val_listfile, args.test_listfile, args.class_map_file, args.repeating_product_file,
                                                               args.input_size, args.batch_size, args.crop_size)
    
    train_loader = dataloaders_dict['train']
    val_loader = dataloaders_dict['val']
    test_loader = dataloaders_dict['test']
    optimizer = set_optimizer(args, classifier)
    # training routine
    for epoch in range(1, args.epochs + 1):
        print("Start training epoch {}".format(epoch))
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc_top1, acc_top5 = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, args)
        time2 = time.time()
        logger.log_value('loss', loss, epoch)
        print('Train epoch {}, total time {:.3f}, accuracy_top1:{:.3f}, accuracy_top5:{:.3f}'.format(
            epoch, time2 - time1, acc_top1, acc_top5))

        # eval for one epoch
        loss, val_acc_top1, val_acc_top5 = validate(val_loader, model, classifier, criterion, args)
        logger.log_value('val_loss', loss, epoch)
        logger.log_value('val_top1', val_acc_top1, epoch)
        if val_acc_top1 > best_acc:
            loss_test, test_acc_top1, test_acc_top5 = test(test_loader, model, classifier, criterion, args)
            best_acc = val_acc_top1
            best_test_acc = test_acc_top1
            output_file = args.save_folder + '/checkpoint_{:04d}.pth.tar'.format(epoch)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'resnet50',
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'classifier': classifier.state_dict(),
            }, is_best=False,
                filename=output_file)
    print('best accuracy: Val {:.3f}, Test {:.3f}'.format(best_acc, best_test_acc))
    return

def set_model(args):
    model = resnet_modified.MyResNet(name='resnet50')
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name='resnet50', num_classes=args.num_classes)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = ckpt['state_dict']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.encoder", "encoder.module")
                if k.startswith("module.head"):
                    k = k.replace("module.head", "head")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        model.encoder = torch.nn.DataParallel(model.encoder)
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        model.load_state_dict(state_dict)

    return model, classifier, criterion

def train(train_loader, model, classifier, criterion, optimizer, epoch, args):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        labels = torch.stack(labels, dim=1)[:, 0]
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)
        top5.update(acc5[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    return losses.avg, top1.avg, top5.avg

def validate(val_loader, model, classifier, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')

    # switch to evaluate mode
    model.eval()
    classifier.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = torch.stack(labels, dim=1)[:, 0]
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)
            top5.update(acc5[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % args.print_freq == 0:
                print('Val: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1, top5=top5))

        # TODO: this should also be done with the ProgressMeter
        print(' * Val Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

def test(test_loader, model, classifier, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')

    # switch to evaluate mode
    model.eval()
    classifier.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(test_loader):
            images = images.float().cuda()
            labels = torch.stack(labels, dim=1)[:, 0]
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)
            top5.update(acc5[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    idx, len(test_loader), batch_time=batch_time,
                    loss=losses, top1=top1, top5=top5))

        # TODO: this should also be done with the ProgressMeter
        print(' * Test Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

def load_data(root_dir, train_listfile, val_listfile, test_listfile, class_map_file,
              class_seen_file, class_unseen_file,
              input_size, scale_size, crop_size, batch_size):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    train_dataset = DatasetCategory(root_dir, 'train', train_listfile,
                                  val_listfile, test_listfile, class_map_file,
                                  class_seen_file, class_unseen_file,
                                  data_transforms['train'])
    val_dataset = DatasetCategory(root_dir, 'val', train_listfile,
                                  val_listfile, test_listfile, class_map_file,
                                  class_seen_file, class_unseen_file,
                                  data_transforms['val'])
    test_dataset = DatasetCategory(root_dir, 'test', train_listfile,
                                   val_listfile, test_listfile, class_map_file,
                                   class_seen_file, class_unseen_file,
                                   data_transforms['test'])
    image_datasets = {'train': train_dataset,
                      'val': val_dataset,
                      'test': test_dataset}

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                       shuffle=False, num_workers=0) for x in ['train', 'val', 'test']}
    print("Finish Datasets and Dataloaders")

    # Detect if we have GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return dataloaders_dict, device


def load_deep_fashion_hierarchical(root_dir, train_list_file, val_list_file, test_list_file, class_map_file, repeating_product_file, input_size, batch_size,crop_size):
    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data_transforms = {
        'train': train_transform,
        'val': train_transform,
        'test': train_transform}

    train_dataset = DeepFashionHierarchihcalDatasetEval(os.path.join(root_dir, train_list_file),
                                                        os.path.join(root_dir, class_map_file),
                                                        os.path.join(root_dir, repeating_product_file),
                                                        transform=data_transforms['train'])
    val_dataset = DeepFashionHierarchihcalDatasetEval(os.path.join(root_dir, val_list_file),
                                                      os.path.join(root_dir, class_map_file),
                                                      os.path.join(root_dir, repeating_product_file),
                                                      transform=data_transforms['val'])
    test_dataset = DeepFashionHierarchihcalDatasetEval(os.path.join(root_dir, test_list_file),
                                                       os.path.join(root_dir, class_map_file),
                                                       os.path.join(root_dir, repeating_product_file),
                                                       transform=data_transforms['test'])
    image_datasets = {'train': train_dataset,
                      'val': val_dataset,
                      'test': test_dataset}
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                       shuffle=False, num_workers=16) for x in ['train', 'val', 'test']}

    # Detect if we have GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return dataloaders_dict, device

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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

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

if __name__ == '__main__':
    main()
