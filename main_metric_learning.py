import multiprocessing
import sys
from collections import OrderedDict
from multiprocessing import Pool

from utils.config import USE_MULTIPROCESS, CONTAMINATION_PERCENTAGE, USE_MNIST, USE_RANDOM, \
    EPSILON, TRAIN_TEST_SPLIT_RATE, USE_SINE, USE_DD_SQUARES, SECOND_DIM
from format_datasets.format_dataset import create_random_dataset, format_mnist_from_distances, create_sine_dataset, \
    create_double_density_squares, format_ml_dataset
from lloc import lloc, create_nd_embedding, get_violated_constraints, count_raw_violated_constraints, predict
from utils.utils import format_arguments, \
    train_test_split, get_num_points, reduce_embedding, merge_embddings, draw_embedding

from utils.read_dataset import load_datasets

from IPython import embed


def create_dataset_from_embedding(embedding, dataset_labels, dataset_name, using="features"):
    filename = f"./datasets/lloc_output/{dataset_name}-from-{using}.csv"
    with open(filename, "w+") as file:
        # write header
        file.write("idx,value,label\n")

        for k, v in embedding.items():
            file.write(f"{k},{v},{dataset_labels[k]}\n")


def train(dataset_features, dataset_labels, dataset_name):
    best_embedding = OrderedDict()
    min_cost = float("inf")

    if USE_MULTIPROCESS:
        cpu_count = multiprocessing.cpu_count()
    else:
        cpu_count = 1

    process_pool = Pool(cpu_count)

    # TODO: change this, format constraints from distances
    # if USE_MNIST:
    #     constraints, num_points = format_mnist_from_distances()
    # elif USE_RANDOM:
    #     constraints, num_points = create_random_dataset()
    # elif USE_SINE:
    #     constraints, num_points = create_sine_dataset()
    # elif USE_DD_SQUARES:
    #     constraints, num_points = create_double_density_squares()
    USING = "labels"
    num_points = len(dataset_labels)

    constraints = format_ml_dataset(dataset_features, dataset_labels, using=USING, subsample_factor=0.3,
                                    dataset_name=dataset_name)

    # Don't split in train and testing, learn with respect to all constraints
    train_constraints, _ = train_test_split(constraints, test_percentage=0)
    process_pool_arguments = format_arguments(train_constraints, num_points, cpu_count)
    responses = process_pool.starmap(lloc, process_pool_arguments)

    best_embedding = reduce_embedding(best_embedding, min_cost, responses)

    create_dataset_from_embedding(best_embedding, dataset_labels, dataset_name, using=USING)


if __name__ == "__main__":
    for x, y, dataset_name in load_datasets():
        print(f"Doing {dataset_name}")
        embedding = train(x, y, dataset_name)
