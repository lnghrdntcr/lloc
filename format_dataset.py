from random import random as rand
from random import randint
from enum import Enum
from struct import unpack
from tqdm import tqdm
import numpy as np
from PIL import Image
from config import USE_DISTANCE, MNIST_COL_SIZE, MNIST_ROW_SIZE, MNIST_SUBSAMPLE_FACTOR, MNIST_MEAN_VALUE_SCALE, \
    MNIST_MIN_CORR_COEFF, MNIST_DIGIT_EXCLUSION_PROBABILITY, STE_NUM_DIGITS, ROE_SAMPLES, CONTAMINATION_PERCENTAGE, \
    BAR_POSITION_OFFSET
from IPython import embed
from random import sample, choice


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
                        row_split[base * 5 + 4]   # bottom_right_row
                    ]
                    next_idx += 1

            if smart_constraints:
                if TripletType.ONE_CLASS_TRIPLET in row_split:
                    new_ds.append([cache[label] for label in images])
                # else:
                #     new_ds.append([cache[label] for label in images])
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

        for i in tqdm(range(n_samples), desc="[MNIST] Read values", position=BAR_POSITION_OFFSET + 1, leave=False):
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
        for i in tqdm(range(n_samples), desc="[MNIST] Read labels", position=BAR_POSITION_OFFSET + 1, leave=False):
            idx = readInt8(f.read(1))
            y.append([0 for i in range(10)])
            y[i][idx] = 1
            y[i] = np.array(y[i]).reshape(10, 1)
        return y


def read_mnist(subsample=True):
    class_distribution = [0 for _ in range(10)]
    real_class_distribution = class_distribution.copy()
    # x_train = read_x_mnist("./datasets/mnist/train-images-idx3-ubyte", normalize=False)
    # y_train = read_y_mnist("./datasets/mnist/train-labels-idx1-ubyte")

    x_test = read_x_mnist("./datasets/mnist/t10k-images-idx3-ubyte", normalize=False)
    y_test = read_y_mnist("./datasets/mnist/t10k-labels-idx1-ubyte")

    indices = set()
    if subsample:
        for _ in range(int(len(x_test) / MNIST_SUBSAMPLE_FACTOR)):
            indices.add(randint(0, len(x_test)))
    else:
        indices = set([i for i in range(len(x_test))])

    new_x_test = []
    new_y_test = []

    for idx in indices:
        new_x_test.append(x_test[idx])
        new_y_test.append(y_test[idx])

    return np.array(new_x_test), np.array(new_y_test), class_distribution


def create_random_dataset(contamination_percentage=CONTAMINATION_PERCENTAGE):

    dataset = [np.random.rand(1, 10) * 1 / 20 for _ in range(ROE_SAMPLES)]
    distance_matrix = np.zeros((len(dataset), len(dataset)))
    constraints = []

    for i in tqdm(range(len(dataset)), desc="Distance Generation: ", leave=False):
        for j in range(len(dataset)):
            distance_matrix[i, j] = np.linalg.norm(dataset[i] - dataset[j], ord=2)

    for i in tqdm(range(len(dataset)), desc="Triplet Generation : ", leave=False):

        # Take 50 Nearest neighbours
        indexed_digits  = [(i, d) for i, d in enumerate(list(np.ravel(distance_matrix[i, :])))]
        closest_indices = [i for i, _ in sorted(indexed_digits, key=lambda x: x[1])][:50]

        for close_index in closest_indices:
            # take the 50 farthest neighbors
            indexed_digits = [(i, d) for i, d in enumerate(list(np.ravel(distance_matrix[i, :])))]
            farthest_indices = [i for i, _ in sorted(indexed_digits, key=lambda x: x[1], reverse=True)][:50]

            for far_index in farthest_indices:
                next = [close_index, far_index]
                if rand() >= contamination_percentage:
                    constraints.append([i, *next])
                else:
                    constraints.append([i, *np.random.permutation(next)])
    subsampled_constraints = sample(constraints, 2 * ROE_SAMPLES ** 2 // 100)

    return subsampled_constraints, ROE_SAMPLES

def format_mnist_from_distances(contamination_percentage=CONTAMINATION_PERCENTAGE):
    """
    This dataset creation complies to the metodology used in
    STOCHASTIC TRIPLET EMBEDDING -> https://lvdmaaten.github.io/ste/Stochastic_Triplet_Embedding_files/PID2449611.pdf
    :return:
    """
    x_test, y_test, class_distribution = read_mnist(subsample=False)

    constraints = []

    # Subsample STE_NUM_DIGITS digits
    subsample_idxs    = sample(range(len(list(x_test))), STE_NUM_DIGITS)
    subsampled_x      = [x_test[idx] for idx in subsample_idxs]
    subsampled_labels = dict([(i, np.argmax(y_test[digit_idx])) for i, digit_idx in enumerate(subsample_idxs)])

    # create distance matrix
    distance_matrix = np.zeros((STE_NUM_DIGITS, STE_NUM_DIGITS))

    for i in tqdm(range(len(subsampled_x)), desc="Distance Generation: ", position=BAR_POSITION_OFFSET + 1, leave=False):
        for j in range(len(subsampled_x)):
            distance_matrix[i, j] = np.linalg.norm(subsampled_x[i] - subsampled_x[j], ord=2)

    for i in tqdm(range(len(subsampled_x)), desc="Triplet Generation : ", position=BAR_POSITION_OFFSET + 2, leave=False):

        # Take 50 Nearest neighbours
        indexed_digits  = [(i, d) for i, d in enumerate(list(np.ravel(distance_matrix[i, :])))]
        closest_indices = [i for i, _ in sorted(indexed_digits, key=lambda x: x[1])][:50]

        for close_index in closest_indices:
            # take the 50 farthest neighbors
            indexed_digits = [(i, d) for i, d in enumerate(list(np.ravel(distance_matrix[i, :])))]
            farthest_indices = [i for i, _ in sorted(indexed_digits, key=lambda x: x[1], reverse=True)][:50]

            for far_index in farthest_indices:
                if rand() >= contamination_percentage:
                    constraints.append([i, close_index, far_index])
                else:
                    constraints.append([i, far_index, close_index])

    # Subsample again reduce the number of constraints constraints
    subsampled_constraints = sample(constraints, STE_NUM_DIGITS ** 2 // 10)
    return subsampled_constraints, STE_NUM_DIGITS


def format_mnist_from_labels(inclusion_probability=1, error_probability=0, use_distance=USE_DISTANCE):
    x_test, y_test, class_distribution = read_mnist()
    constraint_distribution = [0 for _ in range(10)]
    constraints = []
    idx_map = {}
    error_count = 0
    # do it on y_test, because is smaller
    for idx, y in tqdm(enumerate(y_test), desc="[MNIST] Triplet generation from labels -> O(n^3) ", leave=False):
        label = np.argmax(y)
        if use_distance:
            for idx2, y_2 in enumerate(y_test):
                label2 = np.argmax(y_2)
                for idx3, y_3 in enumerate(y_test):
                    label3 = np.argmax(y_3)
                    if (np.abs(label - label2) < np.abs(label - label3)) and (rand() < inclusion_probability):
                        constraint_distribution[label] += 1
                        idx_map[str(idx)] = label
                        idx_map[str(idx2)] = label2
                        idx_map[str(idx3)] = label3

                        if rand() < (1 - error_probability):
                            constraints.append([idx, idx2, idx3])
                        else:
                            constraints.append(np.random.choice([idx, idx2, idx3], 3, replace=False))
                            error_count += 1
        else:
            for idx2, y_2 in enumerate(y_test):
                label2 = np.argmax(y_2)
                if label2 > label:
                    for idx3, y_3 in enumerate(y_test):
                        label3 = np.argmax(y_3)
                        if label3 > label2:
                            # I finally have a triplet!
                            if rand() < inclusion_probability:
                                constraint_distribution[label] += 1
                                idx_map[str(idx)] = label
                                idx_map[str(idx2)] = label2
                                idx_map[str(idx3)] = label3

                                if rand() < (1 - error_probability):
                                    constraints.append([idx, idx2, idx3])
                                else:
                                    constraints.append(np.random.choice([idx, idx2, idx3], 3, replace=False))
                                    error_count += 1

    print(f"Constraints distribution per digit -> {constraint_distribution}\nTotal number of constraints -> {sum(constraint_distribution)}")
    return constraints, idx_map, (x_test, y_test), class_distribution


def format_mnist_from_correlations():
    x_test, y_test = read_mnist()
    new_dataset = []
    idx_map = {}
    correlations = np.corrcoef(x_test.reshape((x_test.shape[0], x_test.shape[1])))
    for idx, row in tqdm(enumerate(correlations), desc="[MNIST] Triplet generation from correlations -> O(n^3)", leave=False):
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
