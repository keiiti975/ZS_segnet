"""Image Folder Data loader
   for zero shot segmentation"""
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import os.path
from tqdm import tqdm

max_size = 640  # unused parameter (max size of image)


def make_dataset(input_dir, target_dir, map_dir, filenames):
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

        if map_dir is not None:
            item.append(os.path.join(map_dir, filename))
        else:
            item.append(None)

        images.append(item)

    return images


def make_vectors(filename):
    """Create semantic_vector array"""
    vector_array = []

    text_file = open(filename, 'r')
    lines = text_file.readlines()
    text_file.close()

    for line in lines:
        line = line.rstrip()
        # input all attribute
        vector1 = line.split(" ")
        # remove index
        vector2 = vector1[1:]
        vector_array.append(vector2)

    vector_array = np.array(vector_array, 'float32')
    return vector_array


class ImageFolderDenseFileLists(data.Dataset):
    """Main Class for Image Folder loader."""

    def __init__(self, input_root='./data/train/...',
                 target_root='./data/train/...', map_root='./data/train/...',
                 filenames='', semantic_filename='./class.txt',
                 training=True, batch_size=1, transform=None):
        """Init function."""
        # get the lists of images
        imgs = make_dataset(input_root, target_root, map_root, filenames)

        # get semantic_vector array
        v_array = make_vectors(semantic_filename)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " +
                               input_root + "\n"))

        self.input_root = input_root
        self.target_root = target_root
        self.map_root = map_root
        self.imgs = imgs
        self.training = training
        self.transform = transform
        self.v_array = v_array

    def __getitem__(self, index):
        """Get item."""

        if self.training is True:
            input_paths = self.imgs[index][0]
            target_paths = self.imgs[index][1]
            map_paths = self.imgs[index][2]

            input_img = self.loader(input_paths)
            target_img = self.loader(target_paths)
            map_img = self.loader(map_paths)

            # apply transformation
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
            map_img = self.transform(map_img)

            # target_img to tensor
            target_img = np.asarray(target_img)
            target_img, mask = self.index2vec(target_img)
            target_img = torch.from_numpy(target_img)

            # map_img to tensor
            target_map = np.asarray(map_img)

            # input_img to tensor
            transform = transforms.Compose([transforms.ToTensor()])
            input_img = transform(input_img)

            data = {'input': input_img, 'target': target_img,
                    'mask': mask, 'map': target_map}

            return data
        else:
            input_paths = self.imgs[index][0]

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

    def padding(self, input_img, target_img, map_img):
        """padding tensor"""
        input_img2 = np.asarray(input_img)
        height = input_img2.shape[0]
        width = input_img2.shape[1]
        transform_input = transforms.Compose(
            [transforms.Pad(
                padding=((max_size - width) // 2,
                         (max_size - height) // 2,
                         (max_size - width) // 2 +
                         (max_size - width) % 2,
                         (max_size - height) // 2 +
                         (max_size - height) % 2,
                         ), fill=0)])
        transform_target = transforms.Compose(
            [transforms.Pad(
                padding=((max_size - width) // 2,
                         (max_size - height) // 2,
                         (max_size - width) // 2 +
                         (max_size - width) % 2,
                         (max_size - height) // 2 +
                         (max_size - height) % 2,
                         ), fill=255)])
        input_img = transform_input(input_img)
        target_img = transform_target(target_img)
        map_img = transform_target(map_img)
        return input_img, target_img, map_img

    def index2vec(self, img):
        """index to semantic vector and return annotations, mask"""
        image = img.copy()
        mask = np.ones(img.shape, dtype='int32')
        smv = np.zeros(self.v_array[0].shape, dtype='float32')
        smv = np.append(smv[None, :], np.ones(
            self.v_array[0].shape, dtype='float32')[None, :], axis=0)
        height = img.shape[0]
        width = img.shape[1]
        for h in range(height):
            for w in range(width):
                index = image[h, w]
                if index > 181:
                    image[h, w] = 182
                    mask[h, w] = 0

        annotation = self.v_array[image]
        mask = smv[mask]
        annotation = annotation.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        return annotation, mask
