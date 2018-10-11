"""zero shot segmentation using segnet"""
# import without torch
import numpy as np
import argparse
import os
from random import shuffle

##########
# import torch
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
##########

# import models
import model.segnet as segnet
import zs_dataset_list as datasets

# input,label data settings
input_nbr = 3  # 入力次元数
label_nbr = 100  # 出力次元数(COCOstuffのセマンティックベクトルの次元数)
imsize = 224

# Training settings
parser = argparse.ArgumentParser(description='ZS_segnet')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--load_path', type=str, default="./model/segnet.pth",
                    help='load_model_path (default: "./model/segnet.pth") ')
parser.add_argument('--save_path', type=str, default="./model/segnet.pth",
                    help='save_model_path (default: "./model/segnet.pth") ')
parser.add_argument('--load', action='store_true', default=False,
                    help='enables load model')
args = parser.parse_args()

# device settings
args.cuda = not args.no_cuda and torch.cuda.is_available()
USE_CUDA = args.cuda

# set the seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Create SegNet model
model = segnet.SegNet(input_nbr, label_nbr)
if USE_CUDA:  # convert to cuda if needed
    model.cuda()
else:
    model.float()
model.eval()
print(model)

# define the optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch, trainloader):
    # set model to train mode
    model.train()

    # update learning rate
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # define a loss
    # 今回の場合背景クラスを考慮しないので重み付けはしない
    if USE_CUDA:
        loss = nn.L1Loss(size_average=False).cuda()
    else:
        loss = nn.L1Loss(size_average=False)

    total_loss = 0

    # iteration over the batches
    for batch_id, data in enumerate(trainloader):
        # make batch tensor and target tensor
        batch_th = Variable(torch.Tensor(data['input']))
        target_th = Variable(torch.Tensor(data['target']))
        if batch_th.size(1) == 1:
            continue

        if USE_CUDA:
            batch_th = batch_th.cuda()
            target_th = target_th.cuda()

        # initialize gradients
        optimizer.zero_grad()

        # predictions
        output = model(batch_th)
        # print("forward propagating ...")

        # shape output and target
        target = target_th.view(-1, target_th.size(2), target_th.size(3))

        # calculate loss
        l_ = loss(output, target.long())
        total_loss += l_.item()
        # backward loss
        l_.backward()
        # print("back propagating ...")
        # optimizer step
        optimizer.step()

        # train conditions
        print("batch_id=%d, filename=%s, loss=%f" % (
            batch_id, trainloader.dataset.get_filename(batch_id)[0],
            l_.item()))

    return total_loss


def test(epoch, testloader):
    model.eval()

    # define a loss
    # 今回の場合背景クラスを考慮しないので重み付けはしない
    if USE_CUDA:
        loss = nn.L1Loss(size_average=False).cuda()
    else:
        loss = nn.L1Loss(size_average=False)

    total_loss = 0

    # iteration over the batches
    for batch_id, data in enumerate(testloader):
        # make batch tensor and target tensor
        batch_th = Variable(torch.Tensor(data['input']))
        target_th = Variable(torch.LongTensor(data['target']))
        if batch_th.size(1) == 1:
            continue

        if USE_CUDA:
            batch_th = batch_th.cuda()
            target_th = target_th.cuda()

        # predictions
        output = model(batch_th)
        # print("forward propagating ...")

        # shape output and target
        target = target_th.view(-1, target_th.size(2), target_th.size(3))

        # calculate loss
        l_ = loss(output, target.long())
        total_loss += l_.item()

        # test conditions
        print("batch_id=%d, filename=%s, loss=%f" %
              (batch_id, testloader.dataset.get_filename(batch_id)[0],
               l_.item()))

    return total_loss


def main():
    # compose transforms
    data_transform = transforms.Compose(
        # [transforms.RandomHorizontalFlip()]
        []
    )

    # load dataset
    trainset = datasets.ImageFolderDenseFileLists(
        input_root='./data/train/input', target_root='./data/train/target',
        filenames='./data/train/names.txt', semantic_filename='./class.txt',
        training=True, transform=data_transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    testset = datasets.ImageFolderDenseFileLists(
        input_root='./data/test/input', target_root='./data/test/target',
        filenames='./data/test/names.txt', semantic_filename='./class.txt',
        training=False, transform=None)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # load model
    if args.load is True:
        model.load_from_filename(args.load_path)

    # train and test
    for epoch in range(1, args.epochs + 1):
        print("epoch:%d" % (epoch))

        # training
        train_loss = train(epoch, trainloader)
        print("train_loss:%f" % (train_loss))

        # validation / test
        # test_loss = test(epoch, testloader)
        # print("test_loss " + str(test_loss))
        print()

    # save model
    torch.save(model.state_dict(), args.save_path)


if __name__ == '__main__':
    main()
