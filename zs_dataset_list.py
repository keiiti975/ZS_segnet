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
    print("makeing dataset ...")

    for filename in tqdm(lines):
        filename = filename.split("\n")[0]
        if training is False:
            with Image.open(os.path.join(input_dir, filename))as image:
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


def make_vectors(filename, config):
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

    if config["cos_similarity"] is True:
        # cos_similarity
        similarity = np.zeros(
            [vector_array.shape[0], vector_array.shape[0]], 'float32')
        for i in range(vector_array.shape[0]):
            for j in range(vector_array.shape[0]):
                similarity[i, j] = cos_sim(
                    vector_array[i, :], vector_array[j, :])
        vector_array = similarity

    if config["jaccard_similarity"] is True:
        # jaccard_similarity
        similarity = np.zeros(
            [vector_array.shape[0], vector_array.shape[0]], 'float32')
        for i in range(vector_array.shape[0]):
            for j in range(vector_array.shape[0]):
                max = 0
                min = 0
                for k in range(vector_array.shape[1]):
                    if vector_array[i, k] > vector_array[j, k]:
                        max += vector_array[i, k]
                        min += vector_array[j, k]
                    else:
                        max += vector_array[j, k]
                        min += vector_array[i, k]
                    similarity[i, j] = min / max
        vector_array = similarity

    if config["PCA"] is True:
        # PCA
        pca = PCA(n_components=config["n_components"])
        pca.fit(vector_array)
        vector_array = pca.fit_transform(vector_array)

    return vector_array


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


class ImageFolderDenseFileLists(data.Dataset):
    """Main Class for Image Folder loader."""

    def __init__(self, input_root,
                 target_root, map_root,
                 filenames, semantic_filename,
                 training, batch_size, config, transform):
        """Init function."""
        if config["model"] is True:
            # get the lists of images
            imgs = make_dataset(input_root, target_root,
                                map_root, filenames, training)
            self.imgs = imgs
            if len(imgs) == 0:
                raise(RuntimeError(
                    "Found 0 images in subfolders of: " + input_root + "\n"))

        # get semantic_vector array
        v_array = make_vectors(semantic_filename, config)
        v_array = torch.from_numpy(v_array)
        v_array = F.normalize(v_array)
        v_array = v_array.numpy()

        self.input_root = input_root
        self.target_root = target_root
        self.map_root = map_root
        self.training = training
        self.batch_size = batch_size
        self.transform = transform
        self.v_array = v_array
        self.config = config

    def __getitem__(self, index):
        """Get item."""

        if self.training is True:
            """train"""
            if self.config["model"] is True:
                """true model"""
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

                # input_img to tensor
                transform = transforms.Compose([transforms.ToTensor()])
                input_img = transform(input_img)

                # target_img to tensor
                target_img = np.asarray(target_img)
                if self.config["decoder"] is True:
                    """model and decoder"""
                    # get mask
                    mask1, mask2 = self.getMask(target_img)
                    target_img = torch.from_numpy(target_img)
                    # map_img to tensor
                    target_map = np.asarray(map_img)
                    data = {'input': input_img, 'target': target_img,
                            'mask1': mask1, 'mask2': mask2, 'map': target_map}
                elif self.config["encoder"] is True:
                    """model and encoder"""
                    target_img = torch.from_numpy(target_img[None, :, :])
                    data = {'input': input_img, 'target': target_img}
                else:
                    """model"""
                    target_img, mask = self.index2vec(target_img)
                    target_img = torch.from_numpy(target_img)
                    # map_img to tensor
                    target_map = np.asarray(map_img)
                    data = {'input': input_img, 'target': target_img,
                            'mask': mask, 'map': target_map}

                return data
            else:
                """false model"""
                if self.config["encoder"] is True:
                    """encoder"""
                    input_map = np.full((256, 256), index)
                    target_vec, mask = self.index2vec(input_map)
                    # input_map to tensor
                    input_map = torch.from_numpy(input_map[None, :, :])
                    input_map = input_map.float()
                    # target_vec to tensor
                    target_vec = torch.from_numpy(target_vec)

                    data = {'input': input_map, 'target': target_vec}
                else:
                    """decoder"""
                    input_map = np.full((256, 256), index)
                    target_map = input_map.copy()
                    input_vec, mask = self.index2vec(input_map)
                    # target_map to tensor
                    target_map = torch.from_numpy(target_map)
                    # input_vec to tensor
                    input_vec = torch.from_numpy(input_vec)

                    data = {'input': input_vec, 'target': target_map}

                return data
        else:
            """test"""
            if self.config["model"] is True:
                """model(test)"""
                input_paths = self.imgs[index][0]
                input_img = self.loader(input_paths)

                # apply transformation
                input_img = self.transform(input_img)

                # input_img to tensor
                transform = transforms.Compose([transforms.ToTensor()])
                input_img = transform(input_img)

                data = {'input': input_img}

                return data
            else:
                if self.config["decoder"] is True:
                    """decoder(test)"""
                    input_vec = np.full((256, 256), index)
                    input_vec, mask = self.index2vec(input_vec)
                    input_vec = torch.from_numpy(input_vec)
                    data = {'input': input_vec}

                return data

    def __len__(self):
        """Length."""
        if self.config["model"] is True:
            return len(self.imgs)
        else:
            return self.v_array.shape[0]

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
        image[img > 181] = 182
        mask[img > 181] = 0
        annotation = self.v_array[image]
        annotation = annotation.transpose(2, 0, 1)
        mask = mask.astype('float32')
        return annotation, mask[None, :, :]

    def getMask(self, img):
        """make mask"""
        mask = np.ones(img.shape, dtype='int32')
        mask[img > 181] = 0
        mask = mask.astype('float32')
        return mask, mask[None, :, :]
