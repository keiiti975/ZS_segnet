"""Image Folder Data loader"""
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import os.path
from tqdm import tqdm

max_size = 640  # max size of COCO


def make_dataset(input_dir, target_dir, filenames):
    """Create the dataset."""
    images = []
    # deal with multiple input

    text_file = open(filenames, 'r')
    lines = text_file.readlines()
    text_file.close()
    print("removing grayscales ...")

    for filename in tqdm(lines):
        filename = filename.split("\n")[0]
        with Image.open(os.path.join(input_dir, filename)) as image:
            if image.mode == "L":
                continue
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
                 filenames='./data/train',
                 training=True, batch_size=1, transform=None):
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
        if batch_size == 1:
            self.padding = False
        else:
            self.padding = True

    def __getitem__(self, index):
        """Get item."""

        input_paths = self.imgs[index][0]
        target_path = self.imgs[index][1]

        input_img = self.loader(input_paths)
        target_img = self.loader(target_path)

        # apply transformation
        input_img = self.transform(input_img)
        target_img = self.transform(target_img)
        if self.training is True self.padding is True:
            input_img2 = np.asarray(input_img)
            height = input_img2.shape[0]
            width = input_img2.shape[1]
            transform_input = transforms.Compose(
                [transforms.Pad(
                    padding=((max_size - width) // 2,
                             (max_size - height) // 2,
                             (max_size - width) // 2 + (max_size - width) % 2,
                             (max_size - height) // 2 +
                             (max_size - height) % 2,
                             ), fill=0)])
            transform_target = transforms.Compose(
                [transforms.Pad(
                    padding=((max_size - width) // 2,
                             (max_size - height) // 2,
                             (max_size - width) // 2 + (max_size - width) % 2,
                             (max_size - height) // 2 +
                             (max_size - height) % 2,
                             ), fill=255)])
            input_img = transform_input(input_img)
            target_img = transform_target(target_img)

        # target_img to tensor
        target_img = np.asarray(target_img)
        target_img = torch.from_numpy(target_img)

        # input_img to tensor
        transform = transforms.Compose([transforms.ToTensor()])
        input_img = transform(input_img)

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
