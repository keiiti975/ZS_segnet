import numpy as np
from PIL import Image
import cv2
from skimage.feature import hog
from skimage import exposure
import os
import os.path
from tqdm import tqdm
from multiprocessing import Pool

input_dir = "/home/tanida/workspace/ZS_segnet/data/train"
output_dir = "/home/tanida/workspace/ZS_segnet/data/train/input_HOG"
filelist = os.path.join(input_dir, "names.txt")
filenames = open(filelist, 'r')
lines = filenames.readlines()
filenames.close()
input_dir = os.path.join(input_dir, "input")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in tqdm(lines):
    filename = filename.split("\n")[0]
    img = cv2.imread(os.path.join(input_dir, filename+".jpg"))
    if len(img.shape) == 2:
        continue
    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    Image.fromarray(np.uint8(hog_image_rescaled*256)).save(os.path.join(output_dir, filename+".png"))