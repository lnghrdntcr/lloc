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
    train_test_split


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
    predict(best_embedding, dataset_name, test_constraints, train_constraints)

    # SECOND DIMENSION!
    new_train_set, _ = get_violated_constraints(best_embedding, train_constraints)

    # Calculate num points from constraints
    point_set = set()
    for constraint in new_train_set:
        for el in constraint:
            point_set.add(el)

    new_num_points = len(point_set)

    second_dim_process_pool_args = format_arguments(new_train_set, new_num_points, cpu_count)
    new_responses = process_pool.starmap(llcc, second_dim_process_pool_args)
    second_dim_best_embedding = reduce_embedding(OrderedDict(), float("inf"), new_responses)

    projected_best_embedding = create_nd_embedding(best_embedding, n_dim=2)
    best_violation_count = count_raw_violated_constraints(projected_best_embedding, train_constraints)
    tmp_embedding = projected_best_embedding.copy()
    tmp_best_violation_count = best_violation_count
    print(f"Original Violates {tmp_best_violation_count} constraints")
    skipped = 0
    for k, new_value in tqdm(second_dim_best_embedding.items(), desc="Merging embeddings"):
        next_embedding = tmp_embedding.copy()
        v1, v2 = next_embedding[k]

        # Skip the mapping if the element is mapped to zero
        if (v2 == new_value) or (v1 == 0):
            skipped += 1
            continue

        next_embedding[k] = (v1, new_value)

        # Count of violated constraints of the new embedding
        new_error_count = count_raw_violated_constraints(next_embedding, train_constraints)
        if new_error_count < tmp_best_violation_count:
            tmp_best_violation_count = new_error_count
            tmp_embedding = next_embedding

    print(f"New Violates {tmp_best_violation_count} constraints")
    print(f"Skipped = {skipped / len(second_dim_best_embedding)}")
    process_pool.close()
    best_embedding = tmp_embedding.copy()
    predict(best_embedding, dataset_name, test_constraints, train_constraints)

    exit(0)


def reduce_embedding(best_embedding, min_cost, responses):
    for embedding, n_violated_constraints in responses:
        if n_violated_constraints < min_cost:
            best_embedding = embedding
            min_cost = n_violated_constraints
    return best_embedding


if __name__ == "__main__":
    if USE_RANDOM:
        dataset_name = "RANDOM_DATASET"
    elif USE_MNIST:
        dataset_name = "MNIST_DATASET"

    #print(f"'{dataset_name}',{EPSILON},{CONTAMINATION_PERCENTAGE},{TRAIN_TEST_SPLIT_RATE}", file=sys.stderr)

    main(dataset_name)
