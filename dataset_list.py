"""Image Folder Data loader for zero shot segmentation"""
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import os
import os.path
from tqdm import tqdm

max_size = 640  # unused parameter (max size of image)


def make_dataset(input_dir, target_dir, map_dir, filenames, training):
    """Create the dataset."""
    images = []
    # deal with multiple input

    text_file = open(filenames, 'r')
    lines = text_file.readlines()
    text_file.close()

    for filename in lines:
        filename = filename.split("\n")[0]
        if training is False:
            with Image.open(os.path.join(input_dir, filename+".jpg"))as image:
                if image.mode == "L":
                    continue
        item = []
        item.append(os.path.join(input_dir, filename+".jpg"))

        if target_dir is not None:
            item.append(os.path.join(target_dir, filename+".png"))
        else:
            item.append(None)

        if map_dir is not None:
            item.append(os.path.join(map_dir, filename+".png"))
        else:
            item.append(None)

        images.append(item)

    return images


class ImageFolderDenseFileLists(data.Dataset):
    """Main Class for Image Folder loader."""

    def __init__(self, input_root,
                 target_root, map_root,
                 filenames, semantic_filename,
                 training, model, config, transform, USE_CUDA):
        """Init function."""
        if config["model"] is True:
            # get the lists of images
            imgs = make_dataset(input_root, target_root,
                                map_root, filenames, training)
            self.imgs = imgs
            if len(imgs) == 0:
                raise(RuntimeError(
                    "Found 0 images in subfolders of: " + input_root + "\n"))

        self.input_root = input_root
        self.target_root = target_root
        self.map_root = map_root
        self.training = training
        self.transform = transform
        self.config = config

    def __getitem__(self, index):
        """Get item."""
        
        if self.training is True:
            """train"""
            # load path
            input_paths = self.imgs[index][0]
            target_paths = self.imgs[index][1]
            map_paths = self.imgs[index][2]
            
            # load image
            input_img = self.loader(input_paths)
            target_img = self.loader(target_paths)
            map_img = self.loader(map_paths)

            # apply transformation
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
            map_img = self.transform(map_img)

            # input_img to tensor
            transform = transforms.Compose([transforms.ToTensor()])
            input_img = transform(input_img)

            # target_img to tensor
            target_img = np.asarray(target_img)
            target_img2 = target_img.copy()
            target_img2[target_img > 181]=182
            target_img2 = torch.from_numpy(target_img2).long()

            # map_img to tensor
            target_map = np.asarray(map_img)
            data = {'input': input_img, 'target': target_img2,
                    'map': target_map}
        else:
            """test"""
            # load path
            input_paths = self.imgs[index][0]
            
            # load image
            input_img = self.loader(input_paths)
            
            # apply transformation
            input_img = self.transform(input_img)
            
            # input_img to tensor
            transform = transforms.Compose([transforms.ToTensor()])
            input_img = transform(input_img)
            
            data = {'input': input_img}
            

        return data

    def __len__(self):
        """Length."""
        return len(self.imgs)

    def get_filename(self, index):
        """Get the filename."""
        return self.imgs[index]

    def image_loader(self, path):
        """Load images."""
        return Image.open(path)

    def loader(self, path):
        """Load Default loader."""
        return self.image_loader(path)
