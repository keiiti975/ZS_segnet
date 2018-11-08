"""zero shot segmentation using segnet"""
# imports without torch
import numpy as np
import argparse
import os
import os.path
import visdom
from PIL import Image
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

# imports Segnet
import model.segnet as segnet
import zs_dataset_list as datasets

# imports utility
import make_log as flog

"""
# imports convnet
import model.convnet as convnet
class_nbr = 256  # 出力次元数(COCOstuffのクラス数)
"""

# input,label data settings
input_nbr = 3  # 入力次元数
label_nbr = 100  # 出力次元数(COCOstuffのセマンティックベクトルの次元数)
imsize = 224

# Training settings
parser = argparse.ArgumentParser(
    description='ZS_segnet,coco can not change batch_size')
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
parser.add_argument('--head', action='store_true', default=False,
                    help='enables head')
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
"""
head = convnet.ConvNet(label_nbr, class_nbr)
"""
if USE_CUDA:  # convert to cuda if needed
    model.cuda()
    """
    head.cuda()
    """
else:
    model.float()
    """
    head.float()
    """
model.eval()
"""
head.eval()
"""
print(model)
"""
print(head)
"""

# Create visdom
vis = visdom.Visdom()

# init window
if args.test is False:
    win = vis.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title='train_loss',
            xlabel='epoch',
            ylabel='loss'
        )
    )
    win_acc = vis.line(
        X=np.array([0]),
        Y=np.array([0]),
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
        loss = nn.MSELoss(size_average=True).cuda()
    else:
        loss = nn.MSELoss(size_average=True)

    total_loss = 0

    # define epoch_size
    epoch_size = len(trainloader)

    # define batch_loss
    batch_loss = 0

    # define annotations
    v_array = trainloader.dataset.v_array
    v_array = torch.from_numpy(v_array)

    # iteration over the batches
    for batch_id, data in enumerate(trainloader):
        # make batch tensor and target tensor
        input = Variable(data['input'])
        target = data['target']
        mask = data['mask']

        if USE_CUDA:
            input = input.cuda()
            target = target.cuda()
            mask = mask.cuda()
            v_array = v_array.cuda()

        # initialize gradients
        optimizer.zero_grad()

        # predictions
        output = model(input)

        # mask tensor
        output = output * mask
        target = target * mask

        # calculate loss
        l_ = 0
        l_ = loss(output, target)
        l_ = l_ * output.size(1)

        total_loss += l_.item()
        # backward loss
        l_.backward()
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

    # define v_array
    v_array = testloader.dataset.v_array
    v_array = torch.from_numpy(v_array)

    # make output_dir
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # iteration over the batches
    for batch_id, data in enumerate(testloader):
        # make batch tensor and target tensor
        input = Variable(data['input'])

        if USE_CUDA:
            input = input.cuda()
            v_array = v_array.cuda()

        # predictions
        output = model(input)

        # output segmentation img
        filename = os.path.basename(
            testloader.dataset.get_filename(batch_id)[0])
        print(filename)
        single_output = output[0, :, :, :]
        single_output = single_output.transpose(0, 1).transpose(1, 2)
        result = min_euclidean(single_output, v_array)
        result = np.uint8(result.cpu().numpy())
        Image.fromarray(result).save(args.output_dir + filename)


def evaluate(output, target_map, v_array, epoch, epoch_size, batch_id):
    target_map = target_map.cpu().numpy()
    data_num = 0
    correct_num = 0
    print("evaluating ...")
    for id in tqdm(range(output.shape[0])):
        single_output = output[id, :, :, :]
        target = target_map[id, :, :]
        single_output = single_output.transpose(0, 1).transpose(1, 2)

        result = min_euclidean(single_output, v_array)
        result = np.uint8(result.cpu().numpy())

        for i in range(output.size(2)):
            for j in range(output.size(3)):
                if target[i, j] < 200:
                    data_num = data_num + 1
                    if result[i, j] == target[i, j]:
                        correct_num = correct_num + 1
    phase = epoch + batch_id / epoch_size
    visualize(phase, (correct_num / data_num), win_acc)


def visualize(phase, visualized_data, window):
    vis.line(
        X=np.array([phase]),
        Y=np.array([visualized_data]),
        update='append',
        win=window
    )


def min_euclidean(out, sem):
    """pytorch calculate euclidean"""
    ab = torch.mm(out.view(-1, label_nbr), sem.t())
    ab = ab.view(out.size(0), out.size(1), sem.size(0))
    aa = (sem**2).sum(1)
    bb = (out**2).sum(-1)
    res = aa[None, None, :] + bb[:, :, None] - 2 * ab
    return res.min(-1)[1]


def min_euclidean2(out, sem):
    """numpy calculate euclidean"""
    ab = np.dot(out, sem.transpose())
    aa = (sem**2).sum(1)
    bb = (out**2).sum(-1)
    res = aa[None, None, :] + bb[:, :, None] - 2 * ab
    return res.argmin(-1)


def main():
    # compose transforms
    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop(256, scale=(1.0, 1.0), ratio=(1.0, 1.0))]
    )
    test_transform = transforms.Compose(
        []
    )

    # load dataset
    trainset = datasets.ImageFolderDenseFileLists(
        input_root='./data/train/input', target_root='./data/train/zs_target2',
        map_root='./data/train/target', filenames='./data/train/names.txt',
        semantic_filename='./v_class/class1.txt', training=True,
        batch_size=args.batch_size, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    testset = datasets.ImageFolderDenseFileLists(
        input_root='./data/test/input', target_root=None,
        map_root=None, filenames='./data/test/names.txt',
        semantic_filename='./v_class/class1.txt', training=False,
        batch_size=1, transform=test_transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=8)

    model.initialized_with_pretrained_weights()

    # load model
    if args.load is True:
        model.load_from_filename("./model/" + args.load_pth)
        """
        head.load_from_filename("./model/head_" + args.load_pth)
        """

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
            """
            torch.save(head.state_dict(),
                       "./model/head_checkpoint_" + str(epoch) + ".pth")
            """
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
        """
        torch.save(head.state_dict(), "./model/head_" + args.save_pth)
        """


if __name__ == '__main__':
    main()
