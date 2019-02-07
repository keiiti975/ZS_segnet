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

tr_map_te = np.asarray([
    26,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    3,
    3,
    3,
    3,
    3,
    3,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    7,
    7,
    7,
    7,
    7,
    7,
    6,
    6,
    6,
    6,
    6,
    6,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    19,
    19,
    15,
    17,
    17,
    15,
    20,
    16,
    25,
    22,
    23,
    23,
    19,
    19,
    14,
    20,
    20,
    19,
    20,
    12,
    20,
    16,
    22,
    22,
    22,
    22,
    22,
    15,
    11,
    18,
    18,
    20,
    15,
    12,
    12,
    13,
    17,
    15,
    20,
    19,
    25,
    20,
    15,
    13,
    12,
    19,
    16,
    25,
    12,
    19,
    15,
    25,
    12,
    12,
    16,
    12,
    11,
    12,
    13,
    17,
    19,
    18,
    12,
    11,
    20,
    14,
    17,
    12,
    13,
    20,
    13,
    15,
    16,
    20,
    17,
    19,
    19,
    15,
    18,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    11,
    11,
    21,
    21,
    13,
    26,
])


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
        item.append(os.path.join(target_dir, filename + ".png"))
        item.append(os.path.join(predict_dir, filename + ".png"))
        images.append(item)

    return images


def main():
    target_dir = './data/test/target'
    predict_dir = os.path.join('./data', args.path)
    filenames = './data/test/names.txt'
    image_path = make_dataset(target_dir, predict_dir, filenames)
    length = len(image_path)
    """ZSL,GZSL
    GT_list = [35, 26, 23, 9, 1, 83, 77, 72, 61, 51, 43, 154, 148,
               149, 105, 123, 112, 127, 152, 167, 109, 179, 116, 102, 175, 99]
    """
    #"""train
    GT_list = [35, 26, 23, 9, 1, 83, 77, 72, 61, 51, 43, 154, 148,
               149, 105, 123, 112, 127, 152, 167, 109, 179, 116, 102, 175, 99, 182]
    #"""
    #"""opt
    opt_list = [31, 27, 10, 73, 57, 50, 46, 45, 128, 106, 97]
    GT_list.extend(opt_list)
    #"""
    GT_num = len(GT_list)
    print("evaluating ...")
    accuracy_train = []
    accuracy_sum = np.zeros(GT_num)
    accuracy_count = np.zeros(GT_num)
    for i in tqdm(range(length)):
        target_img = cv2.imread(image_path[i][0], cv2.IMREAD_GRAYSCALE)
        predict_img = cv2.imread(image_path[i][1], cv2.IMREAD_GRAYSCALE)
        if predict_img is None:
            continue
        """ZSL
        ctarget = target_img.copy()
        target_img[ctarget > 181] = 182
        target_img = tr_map_te[target_img]
        for j in range(GT_num):
            msk = np.isin(ctarget, GT_list[j])
            result = predict_img[msk] == target_img[msk]
            if len(result) != 0:
                acc_te = result.mean()
                accuracy_sum[j] += acc_te
                accuracy_count[j] += 1
        """
        """GZSL
        ctarget = target_img.copy()
        target_img[ctarget > 181] = 182
        for j in range(GT_num):
            msk = np.isin(ctarget, GT_list[j])
            result = predict_img[msk] == target_img[msk]
            if len(result) != 0:
                acc_te = result.mean()
                accuracy_sum[j] += acc_te
                accuracy_count[j] += 1
        """
        #"""train
        msk = np.isin(target_img, GT_list)
        ctarget = target_img.copy()
        target_img[ctarget > 181] = 182
        tr_result = target_img[~msk]==predict_img[~msk]
        if len(tr_result) != 0:
            acc_tr = tr_result.mean()
            accuracy_train.append(acc_tr)
        #"""
    
    #"""train
    acc_tr = np.array(accuracy_train)
    acc1 = np.mean(acc_tr[~np.isnan(acc_tr)])
    print(acc1)
    #"""
    
    """ZSL,GZSL
    Acc = accuracy_sum / accuracy_count

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
    mAcc = 0
    for i in range(GT_num):
        mAcc = mAcc + Acc[i]
        print("%s: Acc=%f" % (GT_list[i], Acc[i]))
        f.write(GT_list[i] + ", Acc=" + str(Acc[i]) + "\n")
    mAcc = mAcc / GT_num
    print("mAcc=%f" % (mAcc))
    f.write("mAcc=" + str(mAcc) + "\n")
    f.close()
    """
    


if __name__ == '__main__':
    main()
