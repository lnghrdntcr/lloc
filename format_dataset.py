from random import random as rand
from enum import Enum
from struct import unpack
from tqdm import tqdm
import numpy as np
from PIL import Image
from config import MNIST_COL_SIZE, MNIST_ROW_SIZE, MNIST_SUBSAMPLE_FACTOR
from IPython import embed


class TripletType(Enum):
    ONE_CLASS_TRIPLET = "ONE_CLASS_TRIPLET"
    TWO_CLASS_TRIPLET = "TWO_CLASS_TRIPLET"
    THREE_CLASS_TRIPLET = "THREE_CLASS_TRIPLET"


def format_google_ds(path, early_stop_count=200):
    with open(path) as ds:
        cache = {}
        reverse_cache = {}
        crop_map = {}
        next_idx = 0

        new_ds = []
        for line in ds:
            if rand() > 0.5:
                continue
            row_split = line.split(",")
            # There is an image every 4 elements
            images = [str(row_split[0]), str(row_split[5]), str(row_split[10])]

            for base, image in enumerate(images):
                if (index := cache.get(image)) is None:
                    cache[image] = next_idx
                    reverse_cache[str(next_idx)] = image
                    crop_map[str(next_idx)] = [
                        row_split[base * 5 + 1],  # top_left_col
                        row_split[base * 5 + 2],  # bottom_right_col
                        row_split[base * 5 + 3],  # top_left_row
                        row_split[base * 5 + 4]  # bottom_right_row
                    ]
                    next_idx += 1

            if TripletType.THREE_CLASS_TRIPLET in row_split:
                new_ds.append([cache[label] for label in reversed(images)])
            else:
                new_ds.append([cache[label] for label in images])

            if next_idx >= early_stop_count:
                break

        return new_ds, reverse_cache, crop_map


def readInt32(num):
    return unpack(">I", num)[0]


def readInt8(num):
    return unpack(">B", num)[0]


def read_x_mnist(x_path, normalize=True):
    with open(x_path, 'rb') as f:
        # skip the first 4 bytes, from the beginning of the file
        f.seek(4, 0)
        n_samples = readInt32(f.read(4))
        n_rows = readInt32(f.read(4))
        n_cols = readInt32(f.read(4))
        x = []

        for i in tqdm(range(n_samples)):
            image = []
            for j in range(n_rows * n_cols):
                image.append(readInt8(f.read(1)))
            x.append(np.array(image).reshape(784, 1))

        ret = np.array(x)

        if normalize:
            return ret / np.max(ret) - 0.5
        else:
            return ret


def read_y_mnist(y_path):
    with open(y_path, 'rb') as f:
        # skip the first 4 bytes, from the beginning of the file
        f.seek(4, 0)
        n_samples = readInt32(f.read(4))

        y = []
        for i in tqdm(range(n_samples)):
            idx = readInt8(f.read(1))
            y.append([0 for i in range(10)])
            y[i][idx] = 1
            y[i] = np.array(y[i]).reshape(10, 1)
        return y


def read_mnist():
    # x_train = read_x_mnist("./datasets/mnist/train-images-idx3-ubyte", normalize=False)
    # y_train = read_y_mnist("./datasets/mnist/train-labels-idx1-ubyte")

    x_test = read_x_mnist("./datasets/mnist/t10k-images-idx3-ubyte", normalize=False)
    y_test = read_y_mnist("./datasets/mnist/t10k-labels-idx1-ubyte")
    #     return x_train, y_train, x_test, y_test
    return x_test[0::MNIST_SUBSAMPLE_FACTOR], y_test[0::MNIST_SUBSAMPLE_FACTOR]


def format_mnist():
    # x_train, y_train, x_test, y_test = read_mnist()
    x_test, y_test = read_mnist()
    new_dataset = []
    idx_map = {}
    # do it on y_test, because is smaller
    print(f"Dataset size -> {len(y_test)}")
    l1_cache = {}
    for idx, y in tqdm(enumerate(y_test), desc="[MNIST] Fast Triplet generation O(10*n^2)"):
        label = np.argmax(y)
        # if cache miss
        if not l1_cache.get(str(label)):
            l1_cache_builder = []
            for idx2, y_2 in enumerate(y_test):
                label2 = np.argmax(y_2)
                if label2 > label:
                    for idx3, y_3 in enumerate(y_test):
                        label3 = np.argmax(y_3)
                        if label3 > label2:
                            # I finally have a triplet!
                            new_dataset.append([idx, idx2, idx3])
                            l1_cache_builder.append([idx, idx2, idx3])
                            idx_map[str(idx)] = label
                            idx_map[str(idx2)] = label2
                            idx_map[str(idx3)] = label3

            l1_cache[str(label)] = l1_cache_builder
        # If cache hit, save the tuples!
        else:
            tuples = l1_cache[str(label)]
            new_dataset.extend(tuples)


    return new_dataset, idx_map, (x_test, y_test)