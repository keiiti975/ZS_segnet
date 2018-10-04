"""Image Folder Data loader"""

import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import os.path

imsize = 224


def make_dataset(input_dir, target_dir, filenames):
    """Create the dataset."""
    images = []
    # deal with multiple input

    text_file = open(filenames, 'r')
    lines = text_file.readlines()

    for filename in lines:
        filename = filename.split("\n")[0]
        item = []
        item.append(os.path.join(input_dir, filename))

        if target_dir is not None:
            item.append(os.path.join(target_dir, filename))
        else:
            item.append(None)

        images.append(item)

    return images


class ImageFolderDenseFileLists(data.Dataset):
    """Main Class for Image Folder loader."""

    def __init__(self, input_root='./data/train/input',
                 target_root='./data/train/target',
                 filenames='./data/train', training=True, transform=None):
        """Init function."""
        # get the lists of images
        imgs = make_dataset(input_root, target_root, filenames)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " +
                               input_root + "\n"))

        self.input_root = input_root
        self.target_root = target_root
        self.imgs = imgs
        self.training = training
        self.transform = transform

    def __getitem__(self, index):
        """Get item."""

        input_paths = self.imgs[index][0]
        target_path = self.imgs[index][1]

        input_img = self.loader(input_paths)

        # apply transformation(without ToTensor(),Resize())
        input_img = self.transform(input_img)
        transform = transforms.Compose(
            [transforms.Resize([imsize, imsize]), transforms.ToTensor()])
        input_img = transform(input_img)

        if self.training:
            target_img = self.loader(target_path)
            target_img = transform(target_img)
        else:
            # この部分がわからない
            target_img = np.array([index])

        target_img = np.array(target_img)
        data = {'input': input_img, 'target': target_img}

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
