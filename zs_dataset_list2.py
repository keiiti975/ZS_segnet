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
import sys

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
GT_list = [35, 26, 23, 9, 1, 83, 77, 72, 61, 51, 43, 154, 148,
           149, 105, 123, 112, 127, 152, 167, 109, 179, 116,
           102, 175, 99]


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
            with Image.open(os.path.join(input_dir, filename + ".jpg"))as image:
                if image.mode == "L":
                    continue
        item = []
        item.append(os.path.join(input_dir, filename + ".jpg"))

        if target_dir is not None:
            item.append(os.path.join(target_dir, filename + ".png"))
        else:
            item.append(None)

        if map_dir is not None:
            item.append(os.path.join(map_dir, filename + ".png"))
        else:
            item.append(None)

        images.append(item)

    return images


def make_vectors(filename, config, model, USE_CUDA):
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

    if config["SSE"] is False:
        """applying semantic_vector transforms (SSE is False)"""
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
    else:
        """create semantic similarity embedding (SSE is True)"""
        if config["model"] is True and config["SSE"] is True:
            model.eval()
            len = vector_array.shape[0]
            for i in range(len):
                input = torch.from_numpy(vector_array[i, :])
                input = input[None, :, None, None]
                if USE_CUDA:
                    input = input.cuda()
                output = model(input)
                output = output[0, :, 0, 0]
                if i == 0:
                    output2 = output[None, :]
                else:
                    output2 = torch.cat([output2, output[None, :]], dim=0)
            if USE_CUDA:
                vector_array = output2.detach().cpu().numpy()
            else:
                vector_array = output2.detach().numpy()
        elif config["decoder"] is True and config["SSE"] is True:
            # remove unlabeled label
            seen_index = [i for i in range(vector_array.shape[0])
                          if i not in GT_list]
            seen_index = np.array(seen_index)
            vector_array = vector_array[seen_index]

    return vector_array


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


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

        # get semantic_vector array
        v_array = make_vectors(semantic_filename, config, model, USE_CUDA)
        v_array = torch.from_numpy(v_array)
        v_array = F.normalize(v_array)
        v_array = v_array.numpy()

        if config["model"] is False and config["decoder"] is True and \
                config["ZSL"] is True:
            # apply seen_index
            seen_index = [i for i in range(
                v_array.shape[0]) if i not in GT_list]
            seen_index = np.array(seen_index)
            seen_tr_map_te = tr_map_te[seen_index]
            v_array = v_array[seen_index]
            self.seen_index = seen_index
            self.seen_tr_map_te = seen_tr_map_te

        self.input_root = input_root
        self.target_root = target_root
        self.map_root = map_root
        self.training = training
        self.transform = transform
        self.v_array = v_array
        self.config = config

    def __getitem__(self, index):
        """Get item."""

        if self.training is True:
            """train"""
            if self.config["model"] is True and self.config["decoder"] is True:
                """model and decoder"""
                if self.config["ZSL"] is True and self.config["SSE"] is False:
                    """ZSL"""
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
                    msk = np.isin(target_img, GT_list)
                    copy_target_img = target_img.copy()
                    copy_target_img[target_img > 181] = 182
                    target_img = tr_map_te[copy_target_img]
                    target_img[msk] = 26
                    target_img = torch.from_numpy(target_img).long()

                    # map_img to tensor
                    target_map = np.asarray(map_img)

                    data = {'input': input_img, 'target': target_img,
                            'map': target_map}
                else:
                    sys.exit("model is not defined")
            elif self.config["model"] is True:
                """model"""
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
                target_img, mask = self.index2vec(target_img)
                target_img = torch.from_numpy(target_img)

                # map_img to tensor
                target_map = np.asarray(map_img)

                data = {'input': input_img, 'target': target_img,
                        'mask': mask, 'map': target_map}
            elif self.config["decoder"] is True:
                """decoder"""
                if self.config["ZSL"] is True:
                    """ZSL"""
                    input_map = np.full((256, 256), index)
                    target_map = input_map.copy()
                    target_map = self.seen_tr_map_te[target_map]
                    input_vec, mask = self.index2vec(input_map)
                    # input_vec to tensor
                    input_vec = torch.from_numpy(input_vec)
                    # target_map to tensor
                    target_map = torch.from_numpy(target_map)

                    data = {'input': input_vec, 'target': target_map}
                else:
                    """normal"""
                    input_map = np.full((256, 256), index)
                    target_map = input_map.copy()
                    input_vec, mask = self.index2vec(input_map)
                    # input_vec to tensor
                    input_vec = torch.from_numpy(input_vec)
                    # target_map to tensor
                    target_map = torch.from_numpy(target_map)

                    data = {'input': input_vec, 'target': target_map}
            else:
                sys.exit("model is not defined")
        else:
            """test"""
            if self.config["model"] is True and self.config["decoder"] is True:
                """model and decoder"""
                if self.config["ZSL"] is True and self.config["SSE"] is False:
                    """ZSL"""
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
                else:
                    sys.exit("model is not defined")
            elif self.config["model"] is True:
                """model"""
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
            elif self.config["decoder"] is True:
                """decoder"""
                if self.config["ZSL"] is True:
                    """ZSL"""
                    input_map = np.full((256, 256), index)
                    target_map = input_map.copy()
                    target_map = tr_map_te[target_map]
                    input_vec, mask = self.index2vec(input_map)
                    # input_vec to tensor
                    input_vec = torch.from_numpy(input_vec)
                    # target_map to tensor
                    target_map = torch.from_numpy(target_map)

                    data = {'input': input_vec, 'target': target_map}
                else:
                    """normal"""
                    input_map = np.full((256, 256), index)
                    target_map = input_map.copy()
                    input_vec, mask = self.index2vec(input_map)
                    # input_vec to tensor
                    input_vec = torch.from_numpy(input_vec)
                    # target_map to tensor
                    target_map = torch.from_numpy(target_map)

                    data = {'input': input_vec, 'target': target_map}
            else:
                sys.exit("model is not defined")

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
        max_size = 640  # unused parameter (max size of image)
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
        mask1 = mask.astype('int64')
        mask2 = mask[None, :, :].astype('float32')
        return mask1, mask2
