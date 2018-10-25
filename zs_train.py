"""zero shot segmentation using segnet"""
# import without torch
import numpy as np
import argparse
import os
from random import shuffle
import visdom
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

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
import zs_dataset_list as datasets

# imports utility
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
parser.add_argument('--output_dir', type=str, default="./data/output/",
                    help='dir of output_image  (default: "./data/output/") ')
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

# Create log model
f_log = flog.make_log(args.batch_size)

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
        loss = nn.L1Loss(size_average=True).cuda()
    else:
        loss = nn.L1Loss(size_average=True)

    total_loss = 0

    # define epoch_size
    epoch_size = len(trainloader)

    # define batch_loss
    batch_loss = 0

    # define annotations
    v_array = trainloader.dataset.v_array

    # iteration over the batches
    for batch_id, data in enumerate(trainloader):
        # make batch tensor and target tensor
        input = Variable(data['input'])
        target = data['target']

        if USE_CUDA:
            input = input.cuda()
            target = target.cuda()

        # initialize gradients
        optimizer.zero_grad()

        # predictions
        output = model(input)
        # print("forward propagating ...")

        # calculate loss
        l_ = 0
        output2 = output.view(output.size(0), output.size(1), -1)
        target2 = target.view(target.size(0), output.size(1), -1)
        output2.transpose(1, 2)
        target2.transpose(1, 2)
        for i in range(args.batch_size):
            output_sample = output2[i, :, :]
            target_sample = target2[i, :, :]
            l_ = l_ + loss(output_sample, target_sample)

        total_loss += l_.item()
        # backward loss
        l_.backward()
        # print("back propagating ...")
        # optimizer step
        optimizer.step()

        # train conditions
        print("epoch=%d, id=%d, loss=%f" % (epoch, batch_id, l_.item()))

        # visualize train condition
        if batch_id % 10 == 0 and batch_id != 0:
            batch_loss = batch_loss + l_.item()
            batch_loss = batch_loss / 10
            # display visdom board
            phase = epoch + batch_id / epoch_size
            visualize(phase, batch_loss, win)
            batch_loss = 0
        else:
            batch_loss = batch_loss + l_.item()
        if batch_id % 1000 == 0 and batch_id != 0:
            target_map = data["map"]
            evaluate(output, target_map, v_array, epoch, epoch_size, batch_id)

    return total_loss


def test(testloader):
    # set model to eval mode
    model.eval()

    # define neighbors
    v_array = testloader.dataset.v_array
    nbrs = NearestNeighbors(
        n_neighbors=1, algorithm='auto').fit(v_array)

    # iteration over the batches
    for batch_id, data in enumerate(testloader):
        # make batch tensor and target tensor
        input = Variable(data['input'])

        if USE_CUDA:
            input = input.cuda()

        # predictions
        output = model(input)
        # print("forward propagating ...")

        # output segmentation img
        print(testloader.dataset.get_filename(batch_id)[0])
        filename = os.path.basename(
            testloader.dataset.get_filename(batch_id)[0])
        target_map = data["map"]
        output = output.cpu().detach().numpy()
        target_map = target_map.cpu().numpy()
        single_output = output[0, :, :, :]
        result = np.zeros(target_map[0, :, :].shape)
        for i in tqdm(range(single_output.shape[1])):
            for j in range(single_output.shape[2]):
                X = single_output[:, i, j]
                X = X[np.newaxis, :]
                distance, indices = nbrs.kneighbors(X)
                if indices[0, 0] == 182:
                    indices[0, 0] = 255
                result[i, j] = indices[0, 0]
        Image.fromarray(np.uint8(result)).save(args.output_dir + filename)


def evaluate(output, target_map, v_array, epoch, epoch_size, batch_id):
    output = output.cpu().detach().numpy()
    target_map = target_map.cpu().numpy()

    for id in range(args.batch_size):
        print("batch=%d" % (id + 1))
        single_output = output[id, :, :, :]
        target = target_map[id, :, :]
        result = np.zeros(target.shape)
        nbrs = NearestNeighbors(
            n_neighbors=1, algorithm='auto').fit(v_array)

        for i in tqdm(range(single_output.shape[1])):
            for j in range(single_output.shape[2]):
                X = single_output[:, i, j]
                X = X[np.newaxis, :]
                distance, indices = nbrs.kneighbors(X)
                if indices[0, 0] == 182:
                    indices[0, 0] = 255
                result[i, j] = indices[0, 0]

        data_num = 0
        correct_num = 0
        for i in range(output.shape[2]):
            for j in range(output.shape[3]):
                data_num = data_num + 1
                if result[i, j] == target[i, j]:
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
        filenames='./data/test/names.txt', semantic_filename='./class.txt',
        training=True, batch_size=args.batch_size, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    testset = datasets.ImageFolderDenseFileLists(
        input_root='./data/test/input', target_root='./data/test/target',
        filenames='./data/test/names.txt', semantic_filename='./class.txt',
        training=False, batch_size=args.batch_size, transform=test_transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=8)

    model.initialized_with_pretrained_weights()

    # load model
    if args.load is True:
        model.load_from_filename("./model/" + args.load_pth)

    # train and test
    for epoch in range(0, args.epochs):
        print("epoch:%d" % (epoch))

        if args.test is False:
            # training
            train_loss = train(epoch, trainloader)
            print("train_loss:%f" % (train_loss))
            # open log_file
            f_log.open()
            # write log_file
            f_log.write(epoch, train_loss)
            # close log_file
            f_log.close()
            # save checkpoint
            torch.save(model.state_dict(),
                       "./model/checkpoint_" + str(epoch) + ".pth")
        elif args.test is True and args.load is True:
            # test
            test(testloader)
            break
        else:
            print('can not test the model!')
            break

        print()

    # save model
    if args.test is False:
        torch.save(model.state_dict(), "./model/" + args.save_pth)


if __name__ == '__main__':
    main()
