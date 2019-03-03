import argparse
import os
import shutil
import time
import math
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.layers.loss import ContrastiveLoss
from cirtorch.datasets.datahelpers import collate_tuples, cid2filename
from cirtorch.datasets.traindataset import TuplesDataset
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.download import download_train, download_test
from cirtorch.utils.whiten import whitenlearn, whitenapply
from cirtorch.utils.evaluate import compute_map_and_print
from cirtorch.utils.general import get_data_root, htime

training_dataset_names = ['retrieval-SfM-120k', 'scores']
test_datasets_names = ['oxford5k,paris6k', 'roxford5k,rparis6k', 'oxford5k,paris6k,roxford5k,rparis6k', 'scores']
test_whiten_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
pool_names = ['mac', 'spoc', 'gem', 'rmac']
loss_names = ['contrastive']
optimizer_names = ['sgd', 'adam']

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

# export directory, training and val datasets, test datasets
parser.add_argument('directory', metavar='DIR',
                    help='destination where trained network should be saved')
parser.add_argument('--training-dataset', '-d', metavar='DATASET', default='retrieval-SfM-120k', choices=training_dataset_names,
                    help='training dataset: ' +
                        ' | '.join(training_dataset_names) +
                        ' (default: retrieval-SfM-120k)')
parser.add_argument('--no-val', dest='val', action='store_false',
                    help='do not run validation')
parser.add_argument('--test-datasets', '-td', metavar='DATASETS', default='oxford5k,paris6k', choices=test_datasets_names,
                    help='comma separated list of test datasets: ' +
                        ' | '.join(test_datasets_names) +
                        ' (default: oxford5k,paris6k)')
parser.add_argument('--test-whiten', metavar='DATASET', default='', choices=test_whiten_names,
                    help='dataset used to learn whitening for testing: ' +
                        ' | '.join(test_whiten_names) +
                        ' (default: None)')

# network architecture and initialization options
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet101', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet101)')
parser.add_argument('--pool', '-p', metavar='POOL', default='gem', choices=pool_names,
                    help='pooling options: ' +
                        ' | '.join(pool_names) +
                        ' (default: gem)')
parser.add_argument('--whitening', '-w', dest='whitening', action='store_true',
                    help='train model with end-to-end whitening')
parser.add_argument('--not-pretrained', dest='pretrained', action='store_false',
                    help='use model with random weights (default: pretrained on imagenet)')
parser.add_argument('--loss', '-l', metavar='LOSS', default='contrastive',
                    choices=loss_names,
                    help='training loss options: ' +
                        ' | '.join(loss_names) +
                        ' (default: contrastive)')
parser.add_argument('--loss-margin', '-lm', metavar='LM', default=0.7, type=float,
                    help='loss margin: (default: 0.7)')

# train/val options specific for image retrieval learning
parser.add_argument('--image-size', default=1024, type=int, metavar='N',
                    help='maximum size of longer image side used for training (default: 1024)')
parser.add_argument('--neg-num', '-nn', default=5, type=int, metavar='N',
                    help='number of negative image per train/val tuple (default: 5)')
parser.add_argument('--query-size', '-qs', default=2000, type=int, metavar='N',
                    help='number of queries randomly drawn per one train epoch (default: 2000)')
parser.add_argument('--pool-size', '-ps', default=20000, type=int, metavar='N',
                    help='size of the pool for hard negative mining (default: 20000)')

# standard train/val options
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help='gpu id used for training (default: 0)')
parser.add_argument('--workers', '-j', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--batch-size', '-b', default=5, type=int, metavar='N',
                    help='number of (q,p,n1,...,nN) tuples in a mini-batch (default: 5)')
parser.add_argument('--optimizer', '-o', metavar='OPTIMIZER', default='adam',
                    choices=optimizer_names,
                    help='optimizer options: ' +
                        ' | '.join(optimizer_names) +
                        ' (default: adam)')
parser.add_argument('--lr', '--learning-rate', default=1e-6, type=float,
                    metavar='LR', help='initial learning rate (default: 1e-6)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='FILENAME',
                    help='name of the latest checkpoint (default: None)')

min_loss = float('inf')

def main():
    global args, min_loss
    args = parser.parse_args()

    # check if test dataset are downloaded
    # and download if they are not
    download_train(get_data_root())
    download_test(get_data_root())

    # create export dir if it doesnt exist
    directory = "{}".format(args.training_dataset)
    directory += "_{}".format(args.arch)
    directory += "_{}".format(args.pool)
    if args.whitening:
        directory += "_whiten"
    if not args.pretrained:
        directory += "_notpretrained"
    directory += "_{}_m{:.2f}".format(args.loss, args.loss_margin)
    directory += "_{}_lr{:.1e}_wd{:.1e}".format(args.optimizer, args.lr, args.weight_decay)
    directory += "_nnum{}_qsize{}_psize{}".format(args.neg_num, args.query_size, args.pool_size)
    directory += "_bsize{}_imsize{}".format(args.batch_size, args.image_size)

    args.directory = os.path.join(args.directory, directory)
    print(">> Creating directory if it does not exist:\n>> '{}'".format(args.directory))
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    # set cuda visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # set random seeds (maybe pass as argument)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    # create model
    if args.pretrained:
        print(">> Using pre-trained model '{}'".format(args.arch))
        model = init_network(model=args.arch, pooling=args.pool, whitening=args.whitening)
    else:
        print(">> Using model from scratch (random weights) '{}'".format(args.arch))
        model = init_network(model=args.arch, pooling=args.pool, whitening=args.whitening, pretrained=False)

    # move network to gpu
    model.cuda()

    # define loss function (criterion) and optimizer
    if args.loss == 'contrastive':
        criterion = ContrastiveLoss(margin=args.loss_margin).cuda()
    else:
        raise(RuntimeError("Loss {} not available!".format(args.loss)))

    # parameters split into features and pool (no weight decay for pooling layer)
    parameters = [
        {'params': model.features.parameters()},
        {'params': model.pool.parameters(), 'lr': args.lr*10, 'weight_decay': 0}
    ]
    if model.whiten is not None:
        parameters.append({'params': model.whiten.parameters()})

    # define optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, args.lr, weight_decay=args.weight_decay)

    # define learning rate decay schedule
    exp_decay = math.exp(-0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        args.resume = os.path.join(args.directory, args.resume)
        if os.path.isfile(args.resume):
            print(">> Loading checkpoint:\n>> '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            min_loss = checkpoint['min_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay, last_epoch=checkpoint['epoch']-1)
        else:
            print(">> No checkpoint found at '{}'".format(args.resume))

    # Data loading code
    normalize = transforms.Normalize(mean=model.meta['mean'], std=model.meta['std'])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = TuplesDataset(
        name=args.training_dataset,
        mode='train',
        imsize=args.image_size,
        nnum=args.neg_num,
        qsize=args.query_size,
        poolsize=args.pool_size,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None,
        drop_last=True, collate_fn=collate_tuples
    )
    if args.val:
        val_dataset = TuplesDataset(
            name=args.training_dataset,
            mode='val',
            imsize=args.image_size,
            nnum=args.neg_num,
            qsize=float('Inf'),
            poolsize=float('Inf'),
            transform=transform
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            drop_last=True, collate_fn=collate_tuples
        )

    # evaluate the network before starting
    test(args.test_datasets, model)

    for epoch in range(start_epoch, args.epochs):

        # set manual seeds per epoch
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)

        # adjust learning rate for each epoch
        scheduler.step()
        # lr_feat = optimizer.param_groups[0]['lr']
        # lr_pool = optimizer.param_groups[1]['lr']
        # print('>> Features lr: {:.2e}; Pooling lr: {:.2e}'.format(lr_feat, lr_pool))

        # train for one epoch on train set
        loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if args.val:
            loss = validate(val_loader, model, criterion, epoch)

        # evaluate on test datasets
        test(args.test_datasets, model)

        # remember best loss and save checkpoint
        is_best = loss < min_loss
        min_loss = min(loss, min_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'meta': model.meta,
            'state_dict': model.state_dict(),
            'min_loss': min_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.directory)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # create tuples for training
    train_loader.dataset.create_epoch_tuples(model)

    # switch to train mode
    model.train()
    model.apply(set_batchnorm_eval)

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # zero out gradients so we can accumulate new ones over batches
        optimizer.zero_grad()

        nq = len(input) # number of training tuples
        ni = len(input[0]) # number of images per tuple

        for q in range(nq):
            output = torch.autograd.Variable(torch.Tensor(model.meta['outputdim'], ni).cuda())
            for imi in range(ni):
                # print(q, imi)
                # target = target.cuda(async=True)
                input_var = torch.autograd.Variable(input[q][imi].cuda())

                # compute output
                output[:, imi] = model(input_var)

            # compute loss for this batch and do backward pass for a batch
            # each backward pass gradients will be accumulated
            target_var = torch.autograd.Variable(target[q].cuda())
            loss = criterion(output, target_var)
            losses.update(loss.data[0])
            loss.backward()

        # do one step for multiple batches
        # accumulated gradients are used
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('>> Train: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch+1, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    return losses.avg


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # create tuples for validation
    val_loader.dataset.create_epoch_tuples(model)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        nq = len(input) # number of training tuples
        ni = len(input[0]) # number of images per tuple
        output = torch.autograd.Variable(torch.Tensor(model.meta['outputdim'], nq*ni).cuda(), volatile=True)

        for q in range(nq):
            for imi in range(ni):
                # target = target.cuda(async=True)
                input_var = torch.autograd.Variable(input[q][imi].cuda())

                # compute output
                output[:, q*ni + imi] = model(input_var)

        target_var = torch.autograd.Variable(torch.cat(target).cuda())
        loss = criterion(output, target_var)

        # record loss
        losses.update(loss.data[0]/nq, nq)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('>> Val: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch+1, i, len(val_loader), batch_time=batch_time, loss=losses))

    return losses.avg

def test(datasets, net):

    print('>> Evaluating network on test datasets...')

    # for testing we use image size of max 1024
    image_size = 1024

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()
    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # compute whitening
    if args.test_whiten:
        start = time.time()

        print('>> {}: Learning whitening...'.format(args.test_whiten))

        # loading db
        db_root = os.path.join(get_data_root(), 'train', args.test_whiten)
        ims_root = os.path.join(db_root, 'ims')
        db_fn = os.path.join(db_root, '{}-whiten.pkl'.format(args.test_whiten))
        with open(db_fn, 'rb') as f:
            db = pickle.load(f)
        images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

        # extract whitening vectors
        print('>> {}: Extracting...'.format(args.test_whiten))
        wvecs = extract_vectors(net, images, image_size, transform)

        # learning whitening
        print('>> {}: Learning...'.format(args.test_whiten))
        wvecs = wvecs.numpy()
        m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
        Lw = {'m': m, 'P': P}

        print('>> {}: elapsed time: {}'.format(args.test_whiten, htime(time.time()-start)))
    else:
        Lw = None

    # evaluate on test datasets
    datasets = args.test_datasets.split(',')
    for dataset in datasets:
        start = time.time()

        print('>> {}: Extracting...'.format(dataset))

        if dataset == "scores":
            # Special added logic to handle loading our score dataset
            from score_retrieval.exports import (
                cfg,
                images,
                qimages,
            )

            print('>> {}: database images...'.format(dataset))
            vecs = extract_vectors(net, images, args.image_size, transform)
            print('>> {}: query images...'.format(dataset))
            qvecs = extract_vectors(net, qimages, args.image_size, transform)

        else:
            # prepare config structure for the test dataset
            cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
            images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
            qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
            bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]

            # extract database and query vectors
            print('>> {}: database images...'.format(dataset))
            vecs = extract_vectors(net, images, image_size, transform)
            print('>> {}: query images...'.format(dataset))
            qvecs = extract_vectors(net, qimages, image_size, transform, bbxs)

        print('>> {}: Evaluating...'.format(dataset))

        # convert to numpy
        vecs = vecs.numpy()
        qvecs = qvecs.numpy()

        # search, rank, and print
        scores = np.dot(vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)
        compute_map_and_print(dataset, ranks, cfg['gnd'])

        if Lw is not None:
            # whiten the vectors
            vecs_lw  = whitenapply(vecs, Lw['m'], Lw['P'])
            qvecs_lw = whitenapply(qvecs, Lw['m'], Lw['P'])

            # search, rank, and print
            scores = np.dot(vecs_lw.T, qvecs_lw)
            ranks = np.argsort(-scores, axis=0)
            compute_map_and_print(dataset + ' + whiten', ranks, cfg['gnd'])

        print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))


def save_checkpoint(state, is_best, directory):
    filename = os.path.join(directory, 'model_epoch%d.pth.tar' % state['epoch'])
    torch.save(state, filename)
    if is_best:
        filename_best = os.path.join(directory, 'model_best.pth.tar')
        shutil.copyfile(filename, filename_best)


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


def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        # freeze running mean and std
        m.eval()
        # freeze parameters
        # for p in m.parameters():
            # p.requires_grad = False


if __name__ == '__main__':
    main()
