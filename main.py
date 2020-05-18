import multiprocessing
import sys
from collections import OrderedDict
from itertools import combinations
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from config import SUPPORTED_DATASETS, USE_MULTIPROCESS, USE_DISTANCE, CONTAMINATION_PERCENTAGE, USE_MNIST, USE_RANDOM, \
    EPSILON, TRAIN_TEST_SPLIT_RATE
from format_dataset import create_random_dataset, format_mnist_from_distances
from llcc import llcc, create_nd_embedding, get_violated_constraints, count_raw_violated_constraints, predict
from utils import setup_results_directories, format_arguments, \
    train_test_split, get_num_points, reduce_embedding, merge_embddings


def main(dataset_name):
    best_embedding = OrderedDict()
    min_cost = float("inf")

    if USE_MULTIPROCESS:
        cpu_count = multiprocessing.cpu_count()
    else:
        cpu_count = 1

    process_pool = Pool(cpu_count)

    # idx_constraints, reverse_cache, (x, y), _ = format_mnist_from_labels()
    # format_google_ds("./datasets/FEC_dataset/faceexp-comparison-data-train-public.csv", smart_constraints=False, early_stop_count=10000)

    if USE_MNIST:
        constraints, num_points = format_mnist_from_distances()
    elif USE_RANDOM:
        constraints, num_points = create_random_dataset()

    train_constraints, test_constraints = train_test_split(constraints, test_percentage=TRAIN_TEST_SPLIT_RATE)
    process_pool_arguments = format_arguments(train_constraints, num_points, cpu_count)
    responses = process_pool.starmap(llcc, process_pool_arguments)

    best_embedding = reduce_embedding(best_embedding, min_cost, responses)
    predict(best_embedding, dataset_name, test_constraints, train_constraints, embedding_dim=1)

    # SECOND DIMENSION!
    new_train_set, _ = get_violated_constraints(best_embedding, train_constraints)

    new_num_points = get_num_points(new_train_set)

    process_pool_args = format_arguments(new_train_set, new_num_points, cpu_count)
    responses = process_pool.starmap(llcc, process_pool_args)
    new_best_embedding = reduce_embedding(OrderedDict(), float("inf"), responses)

    projected_best_embedding = create_nd_embedding(best_embedding, n_dim=2)
    best_violation_count = count_raw_violated_constraints(projected_best_embedding, train_constraints)
    new_embedding = projected_best_embedding.copy()
    new_violation_count = best_violation_count
    print(f"Original Violates {new_violation_count} constraints", file=sys.stderr)
    new_violation_count, best_embedding = merge_embddings(new_best_embedding, new_violation_count,
                                                          new_embedding, train_constraints)

    print(f"New Violates {new_violation_count} constraints", file=sys.stderr)
    process_pool.close()
    predict(best_embedding, dataset_name, test_constraints, train_constraints, embedding_dim=2)

    exit(0)


if __name__ == "__main__":
    if USE_RANDOM:
        dataset_name = "RANDOM_DATASET"
    elif USE_MNIST:
        dataset_name = "MNIST_DATASET"

    print(f"'{dataset_name}',{EPSILON},{CONTAMINATION_PERCENTAGE},{TRAIN_TEST_SPLIT_RATE}", file=sys.stderr)

    main(dataset_name)
