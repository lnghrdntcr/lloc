import os
import sys
from random import random as rand
from random import randint
from enum import Enum
from struct import unpack
from tqdm import tqdm
import numpy as np

from utils.config import USE_DISTANCE, MNIST_SUBSAMPLE_FACTOR, MNIST_MIN_CORR_COEFF, STE_NUM_DIGITS, ROE_SAMPLES, \
    CONTAMINATION_PERCENTAGE, \
    BAR_POSITION_OFFSET
from random import sample as subsample
from random import choice as choice_of


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
                        row_split[base * 5 + 4]  # bottom_right_row
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


def sparsify_instance(subsampled_constraints):
    # Remove half of the constraints in which the first half of the points appear
    new_constraints = []
    point_set = set()
    remove_until_index = len(subsampled_constraints) // 2

    for i, j, k in subsampled_constraints:
        if i < remove_until_index:
            if rand() > 0.5:
                new_constraints.append((i, j, k))
                point_set.update([i, j, k])

    return new_constraints, len(point_set)


def create_random_dataset(contamination_percentage=CONTAMINATION_PERCENTAGE, sparsify=False):
    # First lookup if a dataset with that contamination percentage has already been created
    if os.path.exists(f"./datasets/random/random-{contamination_percentage}.txt"):
        print("Using old dataset", file=sys.stderr)
        with open(f"./datasets/random/random-{contamination_percentage}.txt") as random_ds:
            constraints = []
            for line in random_ds.readlines():
                i, j, k = [int(x) for x in line.replace("\n", "").split(",")]
                constraints.append([i, j, k])

            return constraints, ROE_SAMPLES
    else:
        if os.path.exists(f"./datasets/random/random-0.0.txt"):
            print("Using old dataset", file=sys.stderr)
            with open(f"./datasets/random/random-0.0.txt") as random_ds:
                constraints = []
                for line in random_ds.readlines():
                    i, j, k = [int(x) for x in line.replace("\n", "").split(",")]
                    if rand() > contamination_percentage:
                        constraints.append([i, j, k])
                    else:
                        constraints.append([i, k, j])

            with open(f"./datasets/random/random-{contamination_percentage}.txt", "w+") as random_ds:
                for idx, constraint in enumerate(constraints):
                    i, j, k = constraint
                    if idx != len(constraints) - 1:
                        random_ds.write(f"{i},{j},{k}\n")
                    else:
                        random_ds.write(f"{i},{j},{k}")
            return constraints, ROE_SAMPLES

        else:
            # Create the dataset and ...
            print("Creating dataset", file=sys.stderr)
            dataset = [np.random.rand(1, 10) * 1 / 20 for _ in range(ROE_SAMPLES)]
            distance_matrix = np.zeros((len(dataset), len(dataset)))
            constraints = []

            for i in tqdm(range(len(dataset)), desc="Distance Generation: ", leave=False):
                for j in range(len(dataset)):
                    distance_matrix[i, j] = np.linalg.norm(dataset[i] - dataset[j], ord=2)

            for i in tqdm(range(len(dataset)), desc="Triplet Generation : ", leave=False):

                # Take 50 Nearest neighbours
                indexed_digits = [(i, d) for i, d in enumerate(list(np.ravel(distance_matrix[i, :])))]
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
        subsampled_constraints = subsample(constraints, 3000)

        if sparsify:
            return sparsify_instance(subsampled_constraints)
        # Save it as file!
        with open(f"./datasets/random/random-{contamination_percentage}.txt", "w+") as random_ds:
            for idx, constraint in enumerate(subsampled_constraints):
                i, j, k = constraint
                if idx != len(subsampled_constraints) - 1:
                    random_ds.write(f"{i},{j},{k}\n")
                else:
                    random_ds.write(f"{i},{j},{k}")

        return subsampled_constraints, ROE_SAMPLES


def create_sine_dataset(contamination_percentage=CONTAMINATION_PERCENTAGE):
    # if os.path.exists(f"./datasets/random/random-sin-{}")

    subsampled_constraints = create_or_load(contamination_percentage, "sin",_create_sin)
    return subsampled_constraints, ROE_SAMPLES


def create_or_load(contamination_percentage, name, create_fn):
    # Check if file exists
    if os.path.exists(f"./datasets/random/random-{name}-{contamination_percentage}.txt"):
        print("Using old dataset", file=sys.stderr)
        with open(f"./datasets/random/random-{name}-{contamination_percentage}.txt") as random_ds:
            constraints = []
            for line in random_ds.readlines():
                i, j, k = [int(x) for x in line.replace("\n", "").split(",")]
                constraints.append([i, j, k])
            return constraints
    else:
        # if not, check if base file exists
        if os.path.exists(f"./datasets/random/random-{name}-0.0.txt"):
            with open(f"./datasets/random/random-{name}-0.0.txt") as random_ds:
                constraints = []
                for line in random_ds.readlines():
                    i, j, k = [int(x) for x in line.replace("\n", "").split(",")]

                    if rand() >= contamination_percentage:
                        constraints.append([i, j, k])
                    else:
                        constraints.append([i, k, j])

                with open(f"./datasets/random/random-{name}-{contamination_percentage}.txt", "w+") as random_ds:
                    for idx, constraint in enumerate(constraints):
                        i, j, k = constraint
                        if idx != len(constraints) - 1:
                            random_ds.write(f"{i},{j},{k}\n")
                        else:
                            random_ds.write(f"{i},{j},{k}")
                return constraints
        else:
            # create it
            subsampled_constraints = create_fn(contamination_percentage)

            with open(f"./datasets/random/random-{name}-{contamination_percentage}.txt", "w+") as random_ds:
                for idx, constraint in enumerate(subsampled_constraints):
                    i, j, k = constraint
                    if idx != len(subsampled_constraints) - 1:
                        random_ds.write(f"{i},{j},{k}\n")
                    else:
                        random_ds.write(f"{i},{j},{k}")

            return subsampled_constraints


def _create_sin(contamination_percentage):
    x = np.linspace(0, 1, ROE_SAMPLES)
    y = 3 * np.sin(20 * x) + np.random.rand(ROE_SAMPLES) * 2
    dataset = np.array([x, y])
    distance_matrix = np.zeros((len(dataset[0]), len(dataset[0])))
    for i in tqdm(range(len(distance_matrix)), desc="Distance Generation: ", leave=False):
        for j in range(len(distance_matrix)):
            distance_matrix[i, j] = np.linalg.norm(dataset[:, i] - dataset[:, j], ord=2)
    constraints = format_triplets_from_distance(distance_matrix, poison_perc=contamination_percentage)
    subsampled_constraints = subsample(constraints, 3000)
    return subsampled_constraints


def format_triplets_from_distance(distance_matrix, poison_perc=CONTAMINATION_PERCENTAGE):
    constraints = []
    for i in tqdm(range(len(distance_matrix)), desc="Triplet Generation : ", leave=False):
        for j in range(len(distance_matrix)):
            if j == i:
                continue

            for k in range(len(distance_matrix)):
                if k == i or k == j:
                    continue

                if distance_matrix[i, j] < distance_matrix[i, k]:
                    if rand() > poison_perc:
                        constraints.append([i, j, k])
                    else:
                        constraints.append([i, k, j])

    return constraints


def create_double_density_squares(outer_density=0.3, contamination_perc=CONTAMINATION_PERCENTAGE):
    num_points_inner = int((1 - outer_density) * ROE_SAMPLES * 2)
    num_points_outer = int(outer_density * ROE_SAMPLES * 2)
    n_points = num_points_inner + num_points_outer

    def _create_dd_squares(contamination_percentage):
        num_points_inner = int((1 - outer_density) * ROE_SAMPLES * 2)
        num_points_outer = int(outer_density * ROE_SAMPLES * 2)
        points_outer = np.random.rand(num_points_outer, 2) * 2 - 1
        points_inner = np.random.rand(num_points_inner, 2) - 0.5
        dataset = np.concatenate((points_outer, points_inner))
        n_points = num_points_inner + num_points_outer
        distance_matrix = np.zeros((n_points, n_points))

        constraints = []
        contamination_percentage = 0
        for i in tqdm(range(len(distance_matrix)), desc="Distance Generation: ", leave=False):
            for j in range(len(distance_matrix)):
                distance_matrix[i, j] = np.linalg.norm(dataset[i, :] - dataset[j, :], ord=2)

        constraints = format_triplets_from_distance(distance_matrix, poison_perc=contamination_percentage)
        return subsample(constraints, 3000)

    constraints = create_or_load(contamination_perc, "dd_sq", _create_dd_squares)

    return constraints, n_points


def _create_n_density_squares(contamination_percentage):
    close_to_zero = np.random.rand(int(ROE_SAMPLES / 3), 2) / 4
    mid_from_zero = np.random.rand(int(ROE_SAMPLES / 3), 2) / 3 + 0.5
    far_from_zero = np.random.rand(int(ROE_SAMPLES / 3), 2) / 2 + 1
    dataset = np.concatenate((close_to_zero, mid_from_zero, far_from_zero))
    n_points = ROE_SAMPLES
    distance_matrix = np.zeros((n_points, n_points))

    for i in tqdm(range(len(distance_matrix)), desc="Distance Generation: ", leave=False):
        for j in range(len(distance_matrix)):
            distance_matrix[i, j] = np.linalg.norm(dataset[i, :] - dataset[j, :], ord=2)

    constraints = format_triplets_from_distance(distance_matrix, poison_perc=contamination_percentage)
    subsampled_constraints = subsample(constraints, 3000)
    return subsampled_constraints
def create_n_density_squares(contamination_percentage=CONTAMINATION_PERCENTAGE):

    constraints = create_or_load(contamination_percentage, "n_density", _create_n_density_squares)

    return constraints, ROE_SAMPLES


def format_mnist_from_distances(contamination_percentage=CONTAMINATION_PERCENTAGE):
    """
    This dataset creation complies to the metodology used in
    STOCHASTIC TRIPLET EMBEDDING -> https://lvdmaaten.github.io/ste/Stochastic_Triplet_Embedding_files/PID2449611.pdf
    :return:
    """
    def _gen_mnist(c_per):
        constraints = []

        if os.path.exists("mnist_saved_matrix.npy"):
            distance_matrix = np.load("mnist_saved_matrix.npy")
        else:
            x_test, y_test, class_distribution = read_mnist(subsample=False)

            # Subsample STE_NUM_DIGITS digits
            subsample_idxs = subsample(range(len(list(x_test))), STE_NUM_DIGITS)
            subsampled_x = [x_test[idx] for idx in subsample_idxs]
            subsampled_labels = dict([(i, np.argmax(y_test[digit_idx])) for i, digit_idx in enumerate(subsample_idxs)])

            # create distance matrix
            distance_matrix = np.zeros((STE_NUM_DIGITS, STE_NUM_DIGITS))

            for i in tqdm(range(len(subsampled_x)), desc="Distance Generation: ", position=BAR_POSITION_OFFSET + 1,
                          leave=False):
                for j in range(len(subsampled_x)):
                    distance_matrix[i, j] = np.linalg.norm(subsampled_x[i] - subsampled_x[j], ord=2)

            np.save("mnist_saved_matrix", distance_matrix)

        for i in tqdm(range(STE_NUM_DIGITS), desc="Triplet Generation : ", position=BAR_POSITION_OFFSET + 2,
                      leave=False):

            # Take 50 Nearest neighbours
            indexed_digits = [(i, d) for i, d in enumerate(list(np.ravel(distance_matrix[i, :])))]
            closest_indices = [i for i, _ in sorted(indexed_digits, key=lambda x: x[1])][:50]

            for close_index in closest_indices:
                # take the 50 farthest neighbors
                indexed_digits = [(i, d) for i, d in enumerate(list(np.ravel(distance_matrix[i, :])))]
                farthest_indices = [i for i, _ in sorted(indexed_digits, key=lambda x: x[1], reverse=True)][:50]

                for far_index in farthest_indices:
                    if rand() >= c_per:
                        constraints.append([i, close_index, far_index])
                    else:
                        constraints.append([i, far_index, close_index])

        # Subsample again reduce the number of constraints constraints
        subsampled_constraints = subsample(constraints, 3000)
        return subsampled_constraints

    constraints = create_or_load(contamination_percentage, "mnist", _gen_mnist)
    return constraints, STE_NUM_DIGITS


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

    print(
        f"Constraints distribution per digit -> {constraint_distribution}\nTotal number of constraints -> {sum(constraint_distribution)}")
    return constraints, idx_map, (x_test, y_test), class_distribution


def format_mnist_from_correlations():
    x_test, y_test = read_mnist()
    new_dataset = []
    idx_map = {}
    correlations = np.corrcoef(x_test.reshape((x_test.shape[0], x_test.shape[1])))
    for idx, row in tqdm(enumerate(correlations), desc="[MNIST] Triplet generation from correlations -> O(n^3)",
                         leave=False):
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


def format_ml_dataset(x, y, using="features", dataset_name="None", subsample_factor=0.0):
    constraints = []

    if using == "features":
        distance_matrix = np.zeros((len(x), len(x)))
        for i in tqdm(range(len(x)), desc=f"[{dataset_name.upper()}]Distance Generation: ", leave=False):
            for j in range(len(x)):
                distance_matrix[i, j] = np.linalg.norm(x[i] - x[j], ord=2)

        for i in tqdm(range(len(x)), desc=f"[{dataset_name.upper()}]Triplet Generation : ",
                      position=BAR_POSITION_OFFSET + 2,
                      leave=False):

            # Take 50 Nearest neighbours
            indexed_digits = [(i, d) for i, d in enumerate(list(np.ravel(distance_matrix[i, :])))]
            closest_indices = [i for i, _ in sorted(indexed_digits, key=lambda x: x[1])][:50]

            for close_index in closest_indices:
                # take the 50 farthest neighbors
                indexed_digits = [(i, d) for i, d in enumerate(list(np.ravel(distance_matrix[i, :])))]
                farthest_indices = [i for i, _ in sorted(indexed_digits, key=lambda x: x[1], reverse=True)][:50]

                for far_index in farthest_indices:
                    constraints.append([i, close_index, far_index])

    elif using == "labels":
        for i, el_1 in tqdm(enumerate(y), desc=f"[{dataset_name.upper()}]Triplet Generation : "):
            for j, el_2 in enumerate(y):
                if j != i:
                    for k, el_3 in enumerate(y):
                        if k != i and j != k:
                            close = el_1 == el_2
                            distant = el_1 != el_3
                            if close and distant:
                                constraints.append([i, j, k])

        try:
            constraints = subsample(constraints, len(y) * 70 * 70)
        except:
            pass

    return constraints
