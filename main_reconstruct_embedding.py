import multiprocessing
import sys
from collections import OrderedDict
from multiprocessing import Pool

from format_datasets.format_dataset import create_random_dataset, format_mnist_from_distances, create_sine_dataset, \
    create_double_density_squares, create_n_density_squares
from lloc import lloc, create_nd_embedding, get_violated_constraints, count_raw_violated_constraints, predict
from utils.config import USE_MULTIPROCESS, CONTAMINATION_PERCENTAGE, USE_MNIST, USE_RANDOM, \
    EPSILON, TRAIN_TEST_SPLIT_RATE, USE_SINE, USE_DD_SQUARES, SECOND_DIM, USE_CLUSTERS
from utils.utils import format_arguments, \
    train_test_split, get_num_points, reduce_embedding, merge_embeddings


def main(dataset_name):
    best_embedding = OrderedDict()
    min_cost = float("inf")

    if USE_MULTIPROCESS:
        cpu_count = multiprocessing.cpu_count()
    else:
        cpu_count = 1

    process_pool = Pool(cpu_count)


    # Choosing the dataset to test

    if USE_MNIST:
        constraints, num_points = format_mnist_from_distances()
    elif USE_RANDOM:
        constraints, num_points = create_random_dataset()
    elif USE_SINE:
        constraints, num_points = create_sine_dataset()
    elif USE_DD_SQUARES:
        constraints, num_points = create_double_density_squares()
    elif USE_CLUSTERS:
        constraints, num_points = create_n_density_squares()

    train_constraints, test_constraints = train_test_split(constraints, test_percentage=TRAIN_TEST_SPLIT_RATE)
    process_pool_arguments = format_arguments(train_constraints, num_points, cpu_count)
    responses = process_pool.starmap(lloc, process_pool_arguments)

    best_embedding = reduce_embedding(best_embedding, min_cost, responses)
    best_violation_count = count_raw_violated_constraints(best_embedding, train_constraints)
    predict(best_embedding, dataset_name, test_constraints, train_constraints, best_violation_count, embedding_dim=1)

    if SECOND_DIM:
        # SECOND DIMENSION!
        new_train_set, _ = get_violated_constraints(best_embedding, train_constraints)

        new_num_points = get_num_points(new_train_set)

        process_pool_args = format_arguments(new_train_set, new_num_points, cpu_count)
        responses = process_pool.starmap(lloc, process_pool_args)
        new_best_embedding = reduce_embedding(OrderedDict(), float("inf"), responses)

        projected_best_embedding = create_nd_embedding(best_embedding, n_dim=2)
        best_violation_count = count_raw_violated_constraints(projected_best_embedding, train_constraints)
        new_embedding = projected_best_embedding.copy()
        new_violation_count = best_violation_count
        print(f"Original Violates {new_violation_count} constraints", file=sys.stderr)
        new_violation_count, best_embedding = merge_embeddings(new_best_embedding, new_violation_count,
                                                               new_embedding, train_constraints)

        print(f"New Violates {new_violation_count} constraints", file=sys.stderr)
        process_pool.close()
        predict(best_embedding, dataset_name, test_constraints, train_constraints, new_violation_count, embedding_dim=2)

    exit(0)


if __name__ == "__main__":
    if USE_RANDOM:
        dataset_name = "RANDOM_DATASET"
    elif USE_MNIST:
        dataset_name = "MNIST_DATASET"
    elif USE_SINE:
        dataset_name = "SINE"
    elif USE_DD_SQUARES:
        dataset_name = "DOUBLE_DENSITY_SQUARE"
    elif USE_CLUSTERS:
        dataset_name = "CLUSTERS"

    print(f"'{dataset_name}',{EPSILON},{CONTAMINATION_PERCENTAGE},{TRAIN_TEST_SPLIT_RATE}", file=sys.stderr)

    main(dataset_name)
