import torch
import numpy as np
import os
import os.path
import cv2
from tqdm import tqdm
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description='evaluate ZS_segnet')
parser.add_argument('path', type=str,
                    help='load image path(=project name)  (place: "./data") ')
args = parser.parse_args()


def make_dataset(target_dir, predict_dir, filenames):
    """Create the dataset."""
    images = []
    # deal with multiple input

    text_file = open(filenames, 'r')
    lines = text_file.readlines()
    text_file.close()

    for filename in tqdm(lines):
        filename = filename.split("\n")[0]
        item = []
        item.append(os.path.join(target_dir, filename))
        item.append(os.path.join(predict_dir, filename))
        images.append(item)

    return images


def evaluate(target_img, predict_img, GT_list, GT_num):
    GT_pixel_num = np.zeros(GT_num, dtype='int32')
    predict_pixel_num = np.zeros(GT_num, dtype='int32')
    predict_TP_num = np.zeros(GT_num, dtype='int32')
    GT_root = np.ones(target_img.shape, dtype='int32')

    for i in range(GT_num):
        GT = GT_root * GT_list[i]
        img1 = target_img - GT
        img2 = predict_img - GT
        img1c = img1.copy()
        img2c = img2.copy()
        img1[img1c == 0] = 1
        img1[img1c != 0] = 0
        img2[img2c == 0] = 1
        img2[img2c != 0] = 0
        img3 = img1 + img2
        img3c = img3.copy()
        img3[img3c == 2] = 1
        img3[img3c != 2] = 0
        GT_pixel_num[i] = np.sum(img1)
        predict_pixel_num[i] = np.sum(img2)
        predict_TP_num[i] = np.sum(img3)

    return GT_pixel_num, predict_pixel_num, predict_TP_num


def main():
    target_dir = './data/test/target'
    predict_dir = os.path.join('./data', args.path)
    filenames = './data/test/names.txt'
    image_path = make_dataset(target_dir, predict_dir, filenames)
    length = len(image_path)
    GT_list = [35, 26, 23, 9, 1, 83, 77, 72, 61, 51, 43, 154, 148,
               149, 105, 123, 112, 127, 152, 167, 109, 179, 116, 102, 175, 99]
    GT_num = len(GT_list)
    GT_pixel_all = np.zeros(GT_num, dtype='int32')
    predict_pixel_all = np.zeros(GT_num, dtype='int32')
    predict_TP_all = np.zeros(GT_num, dtype='int32')
    print("evaluating ...")
    for i in tqdm(range(length)):
        target_img = cv2.imread(image_path[i][0], cv2.IMREAD_GRAYSCALE)
        predict_img = cv2.imread(image_path[i][1], cv2.IMREAD_GRAYSCALE)
        if predict_img is None:
            continue
        GT_pixel_num, predict_pixel_num, predict_TP_num = evaluate(
            target_img, predict_img, GT_list, GT_num)
        GT_pixel_all = GT_pixel_all + GT_pixel_num
        predict_pixel_all = predict_pixel_all + predict_pixel_num
        predict_TP_all = predict_TP_all + predict_TP_num

    IoU = predict_TP_all / (GT_pixel_all + predict_pixel_all - predict_TP_all)
    Acc = predict_TP_all / GT_pixel_all

    if not os.path.isdir("./eval"):
        os.makedirs("./eval")
    f = open('./eval/result.txt', mode='a')
    f.write(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "\n")
    f.write(predict_dir + "\n")
    np.set_printoptions(suppress=True)
    GT_list = ["snowboard", "backpack", "zebra", "trafficlight",
               "bicycle", "book", "microwave", "tv",
               "chair", "banana", "bottle", "sea",
               "road", "rock", "clouds", "grass",
               "fence", "house", "salad", "towel",
               "desk", "window_blind", "floor_tile", "ceiling_tile",
               "wall_tile", "cardboard"]
    mIoU = 0
    mAcc = 0
    for i in range(GT_num):
        mIoU = mIoU + IoU[i]
        mAcc = mAcc + Acc[i]
        print("%s: IoU=%f, Acc=%f" % (GT_list[i], IoU[i], Acc[i]))
        f.write(GT_list[i] + ": IoU=" +
                str(IoU[i]) + ", Acc=" + str(Acc[i]) + "\n")
    mIoU = mIoU / GT_num
    mAcc = mAcc / GT_num
    print("mIoU=%f, mAcc=%f" % (mIoU, mAcc))
    f.write("mIoU=" + str(mIoU) + ", mAcc=" + str(mAcc) + "\n")
    f.close()


if __name__ == '__main__':
    main()
