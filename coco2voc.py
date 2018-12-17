import numpy as np
import os
import os.path
from PIL import Image
import cv2
from tqdm import tqdm

input_dir = "./data/MSE_batch12_NZSL"
output_dir = "./data/voc_eval"
input_filenames = open("./data/voc_names.txt", "r")
input_names = input_filenames.readlines()
input_filenames.close()
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

coco_list = [4, 1, 15, 8, 43, 5, 2, 16, 61, 20, 66, 17, 18,
             3, 0, 63, 19, 6, 71]
voc_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 19, 20]

for input_name in tqdm(input_names):
    input_name = input_name.split('\n')[0]
    input_file = os.path.join(input_dir, input_name)
    save_name = input_name.split('.')[0]+'.png'
    input_image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    voc_input_image = np.full(input_image.shape, 255, dtype='uint8')
    for i in range(len(coco_list)):
        voc_input_image[input_image == coco_list[i]] = voc_list[i]
    # print(voc_input_image)
    cv2.imwrite(os.path.join(output_dir, save_name), voc_input_image)
    # Image.fromarray(np.uint8(voc_input_image)).save(os.path.join(output_dir, input_name))
    input_image = cv2.imread(os.path.join(output_dir, save_name),cv2.IMREAD_GRAYSCALE)
    # input_image = np.array(Image.open(os.path.join(output_dir, input_name)))
    print(input_image)
    
