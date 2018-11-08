import torch
import numpy as np
import os
import os.path
import cv2
from tqdm import tqdm
from datetime import datetime

input_dir = open("./data/coco_eval")
input_files = input_dir.readlines()
input_dir.close()
if not os.path.isdir("./data/voc_eval"):
    os.makedirs("./data/voc_eval")

coco_list = [4, 1, 15, 8, 43, 5, 2, 16, 61, 20, 66, 17, 18,
             3, 0, 63, 19, 6, 71]
voc_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 19, 20]

for input_file in input_files:
    input_file = input_file.split('\n')[0]
    input_image = cv2.imread(input_file)
    voc_input_image = np.full(input_image.shape, 255, dtype=int)
    for i in range(coco_list.shape[0]):
        voc_input_image[input_image == coco_list[i]] = voc_list[i]
    cv2.imwrite('./data/voc_eval/' + input_file, np.uint8(voc_input_image))
