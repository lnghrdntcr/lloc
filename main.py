import multiprocessing
from collections import OrderedDict
from itertools import combinations
from multiprocessing import Pool

from tqdm import tqdm

from config import SUPPORTED_DATASETS, MNIST_ERROR_RATE, \
    MNIST_DIGIT_EXCLUSION_PROBABILITY, EPSILON, USE_MULTIPROCESS, USE_DISTANCE
from format_dataset import format_mnist_from_labels
from llcc import llcc
from utils import setup_results_directories, format_arguments, \
    train_test_split
import numpy as np


def main():
    best_embedding = OrderedDict()
    min_cost = float("inf")

    if USE_MULTIPROCESS:
        cpu_count = multiprocessing.cpu_count()
    else:
        cpu_count = 1

    process_pool = Pool(cpu_count)

    idx_constraints, reverse_cache, (x, y), class_distribution = format_mnist_from_labels()
                                                               # format_google_ds("./datasets/FEC_dataset/faceexp-comparison-data-train-public.csv", smart_constraints=False, early_stop_count=10000)

    num_points = len(reverse_cache)
    train_constraints, test_constraints = train_test_split(idx_constraints, test_percentage=0.3)

    process_pool_arguments = format_arguments(train_constraints, num_points, cpu_count)
    responses = process_pool.starmap(llcc, process_pool_arguments)

    for embedding, n_violated_constraints in responses:
        if n_violated_constraints < min_cost:
            best_embedding = embedding
            min_cost = n_violated_constraints

    process_pool.close()

    error_rate = 0
    missing = 0
    for test_constraint in tqdm(test_constraints, desc="Testing..."):
        train_constraints.append(test_constraint)
        new_cost = 0

        if USE_DISTANCE:
            i, j, k = test_constraint
            if ((f_i := best_embedding.get(i)) is not None) and ((f_j := best_embedding.get(j)) is not None) and ((f_k := best_embedding.get(k)) is not None):
                new_cost += int(np.abs(f_i - f_j) > np.abs(f_i - f_k))
            else:
                new_cost += 1
                missing  += 1

            if new_cost != 0:
                error_rate += 1
        else:
            for i, j in list(combinations(test_constraint, 2))[:-1]:

                if ((f_i := best_embedding.get(i)) is not None) and ((f_j := best_embedding.get(j)) is not None):
                    new_cost += int(f_i > f_j)
                else:
                    missing  += 1
                    new_cost += 1

            if new_cost != 0:
                error_rate += 1

        del train_constraints[-1]

    print(f"Accuracy: {100 - (error_rate / len(test_constraints)) * 100}%")
    exit(0)


if __name__ == "__main__":
    # print(
    #     f"Running test with EPSILON={EPSILON}, MNIST_ERROR_RATE={MNIST_ERROR_RATE}, MNIST_DIGIT_EXCLUSION_PROBABILITY={MNIST_DIGIT_EXCLUSION_PROBABILITY}")

    # Build dir structure for the results
    for ds in SUPPORTED_DATASETS:
        setup_results_directories(ds)

    main()
