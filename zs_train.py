"""zero shot segmentation using segnet"""
# import without torch
import numpy as np
import argparse
import os
from random import shuffle
import visdom

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
import make_log as flog

# input,label data settings
input_nbr = 3  # 入力次元数
label_nbr = 100  # 出力次元数(COCOstuffのセマンティックベクトルの次元数)
imsize = 224

# Training settings
parser = argparse.ArgumentParser(
    description='ZS_segnet,coco can not change batch_size')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--load_pth', type=str, default="segnet.pth",
                    help='load pth from ./model (default: "segnet.pth") ')
parser.add_argument('--save_pth', type=str, default="segnet.pth",
                    help='save pth to ./model (default: "segnet.pth") ')
parser.add_argument('--load', action='store_true', default=False,
                    help='enables load model')
parser.add_argument('--test', action='store_true', default=False,
                    help='test the model')
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

# Create visdom
vis = visdom.Visdom()

# init log_data
X = np.array([[0, 0]])
win = vis.scatter(
    X=X,
    opts=dict(
        xlabel='epoch',
        ylabel='loss'
    )
)

# Create log_file
f_log = flog.make_log()

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

    # define epoch_size
    epoch_size = trainloader.dataset.__len__()

    # define batch_loss
    batch_loss = 0

    # iteration over the batches
    for batch_id, data in enumerate(trainloader):
        # make batch tensor and target tensor
        input = Variable(data['input'])
        target = data['target']
        if input.size(1) == 1:
            continue

        if USE_CUDA:
            input = input.cuda()
            target = target.cuda()

        # initialize gradients
        optimizer.zero_grad()

        # predictions
        output = model(input)
        # print("forward propagating ...")

        # shape output and target
        output = output.view(-1, output.size(2), output.size(3))
        target = target.view(-1, target.size(2), target.size(3))

        # calculate loss
        l_ = loss(output, target)
        total_loss += l_.item()
        height = target.size(1)
        width = target.size(2)
        # backward loss
        l_.backward()
        # print("back propagating ...")
        # optimizer step
        optimizer.step()

        # train conditions
        print("epoch=%d, id=%d, filename=%s, loss=%f"
              % (epoch, batch_id,
                 trainloader.dataset.get_filename(batch_id)[0],
                 l_.item() / (height * width)))

        if batch_id % 10 == 0:
            batch_loss = batch_loss + l_.item()
            batch_loss = batch_loss / 10
            # display visdom board
            phase = epoch + batch_id / epoch_size
            X2 = np.array([[phase, batch_loss]])
            vis.scatter(
                X=X2,
                update='append',
                win=win
            )
            batch_loss = 0
        else:
            batch_loss = batch_loss + l_.item()

    return total_loss


def test(testloader):
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
        input = Variable(data['input'])
        target = data['target'].long()
        if input.size(1) == 1:
            continue

        if USE_CUDA:
            input = input.cuda()
            target = target.cuda()

        # predictions
        output = model(input)
        # print("forward propagating ...")

        # shape output and target
        output = output.view(-1, output.size(2), output.size(3))
        target = target.view(-1, target.size(2), target.size(3))

        # calculate loss
        l_ = loss(output, target)
        total_loss += l_.item()
        height = target.size(1)
        width = target.size(2)

        # test conditions
        print("id=%d, filename=%s, loss=%f"
              % (batch_id,
                 testloader.dataset.get_filename(batch_id)[0],
                 l_.item() / (height * width)))

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

    model.initialized_with_pretrained_weights()

    # make log_file
    f_log.open()

    # load model
    if args.load is True:
        model.load_from_filename("./model/" + args.load_pth)

    # train and test
    for epoch in range(0, args.epochs - 1):
        print("epoch:%d" % (epoch))

        if args.test is False:
            # training
            train_loss = train(epoch, trainloader)
            print("train_loss:%f" % (train_loss))
            f_log.write(epoch, train_loss)
        elif args.test is True and args.load is True:
            # test
            test_loss = test(testloader)
            print("test_loss " + str(test_loss))
            break
        else:
            print('can not test the model!')
            break

        print()

    # close log_file
    f_log.close()

    # save model
    torch.save(model.state_dict(), "./model/" + args.save_pth)


if __name__ == '__main__':
    main()
