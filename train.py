# import without torch
import numpy as np
import argparse
import os
from random import shuffle
import visdom

##########
# imports torch
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
##########

# imports models
import model.segnet as segnet
import dataset_list as datasets

# imports utility
import make_log as flog
import batch_dataloader as loader

# input,label data settings
input_nbr = 3  # 入力次元数
label_nbr = 256  # 出力次元数(COCOstuffのラベルの次元数)
imsize = 224

# Training settings
parser = argparse.ArgumentParser(
    description='ZS_segnet,coco can not change batch_size')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)',)
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
                    help='load pth from ./model  (default: "segnet.pth") ')
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

# init window
X = np.array([[0, 0]])
if args.test is False:
    win = vis.scatter(
        X=X,
        opts=dict(
            title='train_loss',
            xlabel='epoch',
            ylabel='loss'
        )
    )
    win_acc = vis.scatter(
        X=X,
        opts=dict(
            title='train_accuracy',
            xlabel='epoch',
            ylabel='accuracy'
        )
    )
else:
    win = vis.scatter(
        X=X,
        opts=dict(
            title='test_loss',
            xlabel='epoch',
            ylabel='loss'
        )
    )

# Create log model
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
        loss = nn.CrossEntropyLoss().cuda()
    else:
        loss = nn.CrossEntropyLoss()

    total_loss = 0

    # define epoch_size
    epoch_size = len(trainloader)

    # define batch_loss
    batch_loss = 0

    # iteration over the batches
    for batch_id, data in enumerate(trainloader):
        # make batch tensor and target tensor
        input = Variable(data['input'])
        target = data['target'].long()

        if USE_CUDA:
            input = input.cuda()
            target = target.cuda()

        # initialize gradients
        optimizer.zero_grad()

        # predictions
        output = model(input)
        # print("forward propagating ...")

        # calculate lossprint(target_img2.shape)
        l_ = loss(output, target)
        total_loss += l_.item()
        # backward loss
        l_.backward()
        # print("back propagating ...")
        # optimizer step
        optimizer.step()

        # train conditions
        print("epoch=%d, id=%d, loss=%f"
              % (epoch, batch_id, l_.item()))

        if batch_id % 10 == 0 and batch_id != 0:
            batch_loss = batch_loss + l_.item()
            batch_loss = batch_loss / 10
            # display visdom board
            phase = batch_id / epoch_size
            visualize(phase, batch_loss, win)
            batch_loss = 0
        else:
            batch_loss = batch_loss + l_.item()
        if batch_id % 100 == 0 and batch_id != 0:
            evaluate(output, target, epoch, epoch_size, batch_id)

    return total_loss


def test(testloader):
    model.eval()

    # define a loss
    # 今回の場合背景クラスを考慮しないので重み付けはしない
    if USE_CUDA:
        loss = nn.CrossEntropyLoss().cuda()
    else:
        loss = nn.CrossEntropyLoss()

    total_loss = 0

    # define epoch_size
    epoch_size = len(testloader)

    # define batch_loss
    batch_loss = 0

    # iteration over the batches
    for batch_id, data in enumerate(testloader):
        # make batch tensor and target tensor
        input = Variable(data['input'])
        target = data['target'].long()

        if USE_CUDA:
            input = input.cuda()
            target = target.cuda()

        # predictions
        output = model(input)
        # print("forward propagating ...")

        # calculate loss
        l_ = loss(output, target)
        total_loss += l_.item()

        # test conditions
        print("id=%d, loss=%f"
              % (batch_id, l_.item()))

        if batch_id % 10 == 0 and batch_id != 0:
            batch_loss = batch_loss + l_.item()
            batch_loss = batch_loss / 10
            # display visdom board
            phase = batch_id / epoch_size
            visualize(phase, batch_loss, win)
            batch_loss = 0
        else:
            batch_loss = batch_loss + l_.item()
        if batch_id % 100 == 0 and batch_id != 0:
            evaluate(output, target, 0, epoch_size, batch_id)

    return total_loss


def evaluate(output, target, epoch, epoch_size, batch_id):
    for id in range(args.batch_size):
        result = output[id, :, :, :]
        result = result.max(0)[1].cpu().numpy()
        r_target = target[id, :, :]
        r_target = r_target.cpu().numpy()
        data_num = 0
        correct_num = 0
        for i in range(output.size(2)):
            for j in range(output.size(3)):
                data_num = data_num + 1
                if result[i, j] == r_target[i, j]:
                    correct_num = correct_num + 1
        phase = epoch + batch_id / epoch_size
        visualize(phase, (correct_num / data_num), win_acc)


def visualize(phase, visualized_data, window):
    X2 = np.array([[phase, visualized_data]])
    vis.scatter(
        X=X2,
        update='append',
        win=window
    )


def main():
    # compose transforms
    train_transform = transforms.Compose(
        # [transforms.RandomHorizontalFlip()]
        []
    )
    test_transform = transforms.Compose(
        # [transforms.RandomHorizontalFlip()]
        []
    )

    # load dataset
    trainset = datasets.ImageFolderDenseFileLists(
        input_root='./data/test/input', target_root='./data/test/target',
        filenames='./data/test/names.txt',
        training=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    testset = datasets.ImageFolderDenseFileLists(
        input_root='./data/test/input', target_root='./data/test/target',
        filenames='./data/test/names.txt',
        training=False, transform=test_transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

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
            print("test_loss:%f " % (test_loss))
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
