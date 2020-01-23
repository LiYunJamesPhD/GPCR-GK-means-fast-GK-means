import argparse
import numpy
import json
import os
import shutil
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
import csv

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Load all model arch available on Pytorch
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/input', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--outf', default='/output',
                    help='folder to output model checkpoints')
parser.add_argument('--evalf', default="/eval" ,help='path to evaluate sample')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--train', action='store_true',
                    help='train the model')
parser.add_argument('--test', action='store_true',
                    help='test a [pre]trained model on new images')
parser.add_argument('-t', '--fine-tuning', action='store_true',
                    help='transfer learning + fine tuning - train only the last FC layer.')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--accuracy', action='store_true',
                    help='show Top-1 and Top-k (or Top-5) accuracy')
parser.add_argument('--loadPreprocessImg', action='store_true', help='flag loading preprocess images')

best_prec1 = torch.FloatTensor([0])

def get_images_label(path):
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        return list(reader)


def get_images_name(folder):
        """Create a generator to list images name at evaluation time"""
        onlyfiles = [f for f in sorted(os.listdir(folder)) if os.path.isfile(os.path.join(folder, f))]
        for f in onlyfiles:
            yield f


def pil_loader(path):
    """Load images from /eval/ subfolder and coverted it to RGB color space"""
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
            return img


def main():
    global args, best_prec1, cuda, labels
    args = parser.parse_args()

    try:
        os.makedirs(args.outf)

    except OSError:
        pass

    # can we use CUDA?
    cuda = torch.cuda.is_available()  # False
    print ("=> using cuda: {cuda}".format(cuda=cuda))

    # Distributed Training?
    args.distributed = args.world_size > 1
    print ("=> distributed training: {dist}".format(dist=args.distributed))

    ############ DATA PREPROCESSING ############
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    testdir = args.evalf

    # Normalize on RGB Value
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Size on model
    if args.arch.startswith('inception'):
        size = (299, 299)
    else:
        size = (224, 224)

    # Train -> Preprocessing -> Tensor
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(size[0]), # 224 or 299
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    # Get number of labels
    labels = len(train_dataset.classes)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    # Pin memory
    if cuda:
        pin_memory = True
    else:
        pin_memory = False

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=pin_memory, sampler=train_sampler)

    # Validate -> Preprocessing -> Tensor
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(299), # 299
            transforms.CenterCrop(size[0]), # 224 or 299
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin_memory)

    if args.test:
        if args.loadPreprocessImg:
            test_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(testdir, transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]), loader=pil_loader),
                batch_size=1, shuffle=False,
                num_workers=args.workers, pin_memory=pin_memory)
        else:
            # Testing -> Preprocessing -> Tensor
            test_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(testdir, transforms.Compose([
                    transforms.Resize(299), # 299
                    transforms.CenterCrop(size[0]), # 224 or 299
                    transforms.ToTensor(),
                    normalize,
                ]), loader=pil_loader),
                batch_size=1, shuffle=False,
                num_workers=args.workers, pin_memory=pin_memory)

    ############ BUILD MODEL ############
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # Create model from scratch or use a pretrained one
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=labels)

    # Freeze model, train only the last FC layer for the transfered task
    if args.fine_tuning:
        print("=> transfer-learning mode + fine-tuning (train only the last FC layer)")
        # Freeze Previous Layers(now we are using them as features extractor)
        for param in model.parameters():
            param.requires_grad = False
        # Fine Tuning the last Layer For the new task
        # RESNET
        if args.arch == 'resnet18':
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, labels)
            parameters = model.fc.parameters()
        # ALEXNET & VGG
        elif args.arch == 'alexnet' or args.arch == 'vgg19':
            model.classifier._modules['6'] = nn.Linear(4096, labels)
            parameters = model.classifier._modules['6'].parameters()
        elif args.arch == 'densenet121': # DENSENET
            model.classifier = nn.Linear(1024, labels)
            parameters = model.classifier.parameters()
        # INCEPTION
        elif args.arch == 'inception_v3':
            # Auxiliary Fc layer
            num_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs, labels)
            # Last layer
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, labels)
            parameters = model.fc.parameters()
        else:
            print("Error: Fine-tuning is not supported on this architecture.")
            exit(-1)
    else:
        parameters = model.parameters()

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if cuda:
       criterion.cuda()

    # Set SGD + Momentum
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if cuda:
                checkpoint = torch.load(args.resume)
            else:
                # Load GPU model on CPU
                checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Load model on GPU or CPU
    if cuda:
        model.cuda()
    else:
        model.cpu()
    ############ TRAIN/EVAL/TEST ############
    cudnn.benchmark = True
    # Evaluate?
    if args.evaluate:
        print("=> evaluating...")
        validate(val_loader, model, criterion)
        return

    # Testing?
    if args.test:
        print("=> testing...")
        # Name generator
        names = get_images_name(os.path.join(testdir, 'images'))
        test(test_loader, model, names, train_dataset.classes, testdir)
        return

    # Training
    if args.train:
        print("=> training...")
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch)

            # Train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)

            # Evaluate on validation set
            prec1 = validate(val_loader, model, criterion)
            # print (prec1)

            # Remember best prec@1 and save checkpoint
            if cuda:
                prec1 = prec1.cpu() # Load on CPU if CUDA
            # Get bool not ByteTensor
            is_best = bool(prec1.numpy() > best_prec1.numpy())
            # Get greater Tensor
            best_prec1 = torch.FloatTensor(max(prec1.numpy(), best_prec1.numpy()))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    """Train the model on Training Set"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        if cuda:
            input, target = input.cuda(async=True), target.cuda(async=True)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        # For nets that have multiple outputs such as Inception
        if isinstance(output, tuple):
            loss = sum((criterion(o, target_var) for o in output))
            # print (output)
            for o in output:
                prec1 = accuracy(o.data, target, topk=(1,))
                top1.update(prec1[0], input.size(0))
            losses.update(loss.item(), input.size(0)*len(output))
        else:
            loss = criterion(output, target_var)
            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0], input.size(0))
            losses.update(loss.data[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Info log every args.print_freq
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1_val} ({top1_avg})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   top1_val=numpy.asscalar(top1.val.cpu().numpy()),
                   top1_avg=numpy.asscalar(top1.avg.cpu().numpy())))


def validate(val_loader, model, criterion):
    """Validate the model on Validation Set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # Evaluate all the validation set
    for i, (input, target) in enumerate(val_loader):
        if cuda:
            input, target = input.cuda(async=True), target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        # compute output
        output = model(input_var)
        # For nets that have multiple outputs such as Inception
        if isinstance(output, tuple):
            loss = sum((criterion(o,target_var) for o in output))
            # print (output)
            for o in output:
                prec1 = accuracy(o.data, target, topk=(1,))
                top1.update(prec1[0], input.size(0))
            losses.update(loss.data[0], input.size(0)*len(output))
        else:
            loss = criterion(output, target_var)
            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0], input.size(0))
            losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Info log every args.print_freq
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1_val} ({top1_avg})'.format(
                   i, len(val_loader), batch_time=batch_time,
                   loss=losses,
                   top1_val=numpy.asscalar(top1.val.cpu().numpy()),
                   top1_avg=numpy.asscalar(top1.avg.cpu().numpy())))

    print(' * Prec@1 {top1}'
          .format(top1=numpy.asscalar(top1.avg.cpu().numpy())))
    return top1.avg


def test(test_loader, model, names, classes, dir_path):
    """Test the model on the Evaluation Folder
    Args:
        - classes: is a list with the class name
        - names: is a generator to retrieve the filename that is classified
    """ 

    # load a json file for Imagenet to get actual labels
    
    with open('imagenet_class_index.json', encoding='utf-8') as data_file:
        imagenet_labels = json.loads(data_file.read())

    pred_labels = {}
    # switch to evaluate mode
    model.eval()
    # Evaluate all the validation set
    for i, (input,_) in enumerate(test_loader):   
        if cuda:
            input = input.cuda(async=True)
        input_var = torch.autograd.Variable(input)

        # compute output
        output = model(input_var)
        # Take last layer output
        if isinstance(output, tuple):
            output = output[len(output)-1]

        k_val = 0
        if labels >= 5:
            k_val = 5
        else:
            k_val = labels
        # get top-k classes
        pred_vals, pred_index = output.topk(k_val, 1, True, True)
        # dim = 1 for summing across columns.
        prob_values = F.softmax(pred_vals, dim = 1).data.cpu().numpy()[0]
        #print('tmp', prob_values) 

        file_name = next(names)
        print ("Images: " + file_name)
        labs_ind = pred_index.cpu().numpy()[0]
        tmp_list = []
        prob_index = 0
        for label_ind in labs_ind:
            # if the label format is Imagenet, then we would convert the labels to actual lables
            if classes[0][0] == 'n' and classes[0][1:].isdigit():
                print(imagenet_labels[str(label_ind)][1],'{:.10f}'.format(prob_values[prob_index]))
                tmp_list.append(imagenet_labels[str(label_ind)][1])
            else: 
                print(classes[label_ind],'{:.10f}'.format(prob_values[prob_index]))
                tmp_list.append(classes[label_ind])
            prob_index += 1
        pred_labels[file_name] = tmp_list
        print('==============')
 
    # Top-1 and Top-k accuracy
    if args.accuracy:
        top_1_top_k_accuracy(dir_path, pred_labels)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(args.outf, filename))
    if is_best:
        shutil.copyfile(os.path.join(args.outf, filename), os.path.join(args.outf,'model_best.pth.tar'))


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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

def top_1_top_k_accuracy(dir_path, pred_labels):
    acutal_img_labels = get_images_label(os.path.join(dir_path, 'test_labels'))
    special_dict = {'_':  r'',
                    '-':  r'',
                    ' ':  r''}
    
    top_1_classified_correct = 0
    top_k_classified_correct = 0

    # calculate TOp-1 and Top-k accuracy
    for one_file in acutal_img_labels:
        # iterate through all predicted labels
        for top_k_label_ind in range(len(pred_labels[one_file[0]])):
            label_collect = [x.strip() for x in pred_labels[one_file[0]][top_k_label_ind].split(',')]
            # loop through every sub-label within one class
            for label_index in range(len(label_collect)):
                if top_k_label_ind == 0:
                    if label_collect[label_index].lower().translate(str.maketrans(special_dict)) == \
                       one_file[1].lower().translate(str.maketrans(special_dict)):
                        top_1_classified_correct += 1
                        top_k_classified_correct += 1
                        break
                else:
                    if label_collect[label_index].lower().translate(str.maketrans(special_dict)) == \
                       one_file[1].lower().translate(str.maketrans(special_dict)):
                        top_k_classified_correct += 1
                        break
    # ========================================
    print('=======================')
    print('Top-1 accuracy: {top1_acc:.2f} %'.format(top1_acc=top_1_classified_correct / float(len(acutal_img_labels)) * 100.0))
    print('Top-k accuracy: {topk_acc:.2f} %'.format(topk_acc=top_k_classified_correct / float(len(acutal_img_labels)) * 100.0))

if __name__ == '__main__':
    main()

