"""Image Folder Data loader
   for zero shot segmentation"""
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
    text_file.close()

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


def make_vectors(filename):
    """Create semantic_vector array"""
    vector_array = []
    vector_array = np.array(vector_array)

    text_file = open(filename, 'r')
    lines = text_file.readlines()
    text_file.close()

    for line in lines:
        line = line.rstrip()
        # input all attribute
        vector1 = line.split(" ")
        # remove index
        vector2 = vector1[1:]
        vector2 = np.array(vector2)
        np.append(vector_array, vector2)

    print(vector_array.type)
    return vector_array


class ImageFolderDenseFileLists(data.Dataset):
    """Main Class for Image Folder loader."""

    def __init__(self, input_root='./data/train/input',
                 target_root='./data/train/target',
                 filenames='', semantic_filename='./class.txt',
                 training=True, transform=None):
        """Init function."""
        # get the lists of images
        imgs = make_dataset(input_root, target_root, filenames)

        # get semantic_vector array
        v_array = make_vectors(semantic_filename)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " +
                               input_root + "\n"))

        self.input_root = input_root
        self.target_root = target_root
        self.imgs = imgs
        self.training = training
        self.transform = transform
        self.v_array = v_array

    def __getitem__(self, index):
        """Get item."""

        input_paths = self.imgs[index][0]
        target_path = self.imgs[index][1]

        input_img = self.loader(input_paths)
        target_img = self.loader(target_path)

        # apply transformation
        input_img = self.transform(input_img)
        target_img = self.transform(target_img)
        target_img = np.asarray(target_img)
        target_img.flags.writeable = True
        target_img = self.index2vec(target_img)
        print(target_img.shape)

        transform = transforms.Compose([transforms.ToTensor()])
        input_img = transform(input_img)
        target_img = transform(target_img)

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

    def index2vec(self, img):
        """index to semantic vector"""
        annotation = []
        annotation = np.array(annotation)
        index = 0
        width = img.shape[1]
        height = img.shape[0]
        for h in range(height):
            list = []
            list = np.array(list)
            for w in range(width):
                index = img[h, w]
                if index > 181:
                    index = 182

                np.append(list, self.v_array[index])
            np.append(annotation, list)

        return annotation
