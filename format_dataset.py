from random import random as rand
from random import randint
from enum import Enum
from struct import unpack
from tqdm import tqdm
import numpy as np
from PIL import Image
from config import MNIST_COL_SIZE, MNIST_ROW_SIZE, MNIST_SUBSAMPLE_FACTOR, MNIST_MEAN_VALUE_SCALE, MNIST_MIN_CORR_COEFF, MNIST_DIGIT_EXCLUSION_PROBABILITY
from IPython import embed


class TripletType(Enum):
    ONE_CLASS_TRIPLET = "ONE_CLASS_TRIPLET"
    TWO_CLASS_TRIPLET = "TWO_CLASS_TRIPLET"
    THREE_CLASS_TRIPLET = "THREE_CLASS_TRIPLET"


def format_google_ds(path, early_stop_count=1000, smart_constraints=False):
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
                index = cache.get(image)
                if index is None:
                    cache[image] = next_idx
                    reverse_cache[str(next_idx)] = image
                    crop_map[str(next_idx)] = [
                        row_split[base * 5 + 1],  # top_left_col
                        row_split[base * 5 + 2],  # bottom_right_col
                        row_split[base * 5 + 3],  # top_left_row
                        row_split[base * 5 + 4]  # bottom_right_row
                    ]
                    next_idx += 1

            if smart_constraints:
                if TripletType.THREE_CLASS_TRIPLET in row_split:
                    new_ds.append([cache[label] for label in reversed(images)])
                else:
                    new_ds.append([cache[label] for label in images])
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

        for i in tqdm(range(n_samples), desc="[MNIST] Read values"):
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
        for i in tqdm(range(n_samples), desc="[MNIST] Read labels"):
            idx = readInt8(f.read(1))
            y.append([0 for i in range(10)])
            y[i][idx] = 1
            y[i] = np.array(y[i]).reshape(10, 1)
        return y


def read_mnist():
    class_distribution = [0 for _ in range(10)]

    # x_train = read_x_mnist("./datasets/mnist/train-images-idx3-ubyte", normalize=False)
    # y_train = read_y_mnist("./datasets/mnist/train-labels-idx1-ubyte")

    x_test = read_x_mnist("./datasets/mnist/t10k-images-idx3-ubyte", normalize=False)
    y_test = read_y_mnist("./datasets/mnist/t10k-labels-idx1-ubyte")

    indices = set()
    for _ in range(int(len(x_test) / MNIST_SUBSAMPLE_FACTOR)):
        indices.add(randint(0, len(x_test)))

    new_x_test = []
    new_y_test = []

    for idx in indices:
        new_x_test.append(x_test[idx])
        new_y_test.append(y_test[idx])

    # Test against pagerank
    sliced_y_test = []
    for digit in new_y_test:
        digit_class = np.argmax(digit)
        if digit_class < 5 and rand() < MNIST_DIGIT_EXCLUSION_PROBABILITY:
            continue
        else:
            sliced_y_test.append(digit)
            class_distribution[int(digit_class)] += 1

    class_distribution = [el / len(sliced_y_test) for el in class_distribution]

    return np.array(new_x_test), np.array(sliced_y_test), class_distribution


def format_mnist_from_labels(inclusion_probability=1, error_probability=0):
    x_test, y_test, class_distribution = read_mnist()
    new_dataset = []
    idx_map = {}
    error_count = 0
    # do it on y_test, because is smaller
    for idx, y in tqdm(enumerate(y_test), desc="[MNIST] Triplet generation from labels -> O(n^3) "):
        label = np.argmax(y)
        for idx2, y_2 in enumerate(y_test):
            label2 = np.argmax(y_2)
            if label2 > label:
                for idx3, y_3 in enumerate(y_test):
                    label3 = np.argmax(y_3)
                    if label3 > label2:
                        # I finally have a triplet!
                        if rand() < inclusion_probability:
                            idx_map[str(idx)] = label
                            idx_map[str(idx2)] = label2
                            idx_map[str(idx3)] = label3

                            if rand() < (1 - error_probability):
                                new_dataset.append([idx, idx2, idx3])
                            else:
                                new_dataset.append(np.random.choice([idx, idx2, idx3], 3, replace=False))
                                error_count += 1
    return new_dataset, idx_map, (x_test, y_test), class_distribution


def format_mnist_from_correlations():
    x_test, y_test = read_mnist()
    new_dataset = []
    idx_map = {}
    correlations = np.corrcoef(x_test.reshape((x_test.shape[0], x_test.shape[1])))
    for idx, row in tqdm(enumerate(correlations), desc="[MNIST] Triplet generation from correlations -> O(n^3)"):
        treshold_value = MNIST_MIN_CORR_COEFF
        for idx2, el in enumerate(row):
            if el >= treshold_value:
                pivot_row = correlations[idx2]
                for idx3, third in enumerate(pivot_row):
                    if third >= treshold_value and (idx != idx2 and idx != idx3 and idx2 != idx3):
                        new_dataset.append([idx, idx2, idx3])
                        idx_map[str(idx)] = np.argmax(y_test[idx])
                        idx_map[str(idx2)] = np.argmax(y_test[idx2])
                        idx_map[str(idx3)] = np.argmax(y_test[idx3])

    return new_dataset, idx_map, (x_test, y_test)
