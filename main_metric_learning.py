import multiprocessing
import sys
from collections import OrderedDict
from multiprocessing import Pool
from tqdm import tqdm

from utils.config import USING, USE_MULTIPROCESS, CONTAMINATION_PERCENTAGE, USE_MNIST, USE_RANDOM, \
    EPSILON, TRAIN_TEST_SPLIT_RATE, USE_SINE, USE_DD_SQUARES, SECOND_DIM
from format_datasets.format_dataset import create_random_dataset, format_mnist_from_distances, create_sine_dataset, \
    create_double_density_squares, format_ml_dataset
from lloc import lloc, create_nd_embedding, get_violated_constraints, count_raw_violated_constraints, predict
from utils.utils import format_arguments, \
    train_test_split, get_num_points, reduce_embedding, merge_embddings, draw_embedding

from utils.read_dataset import load_dataset, load_poisoned


def create_dataset_from_embedding(embedding, dataset_labels, dataset_name, using="features"):
    filename = f"./datasets/lloc_output/{dataset_name}-from-{using}.csv"
    with open(filename, "w+") as file:
        # write header
        file.write("idx,value,label\n")

        for i in tqdm(range(int(1 / EPSILON)), desc=f"[{dataset_name.upper()}] Writing new embedding"):

            points_to_i = [k for k, v in embedding.items() if v == i]
            if len(points_to_i) == 0:
                continue
            spacing = 1 / len(points_to_i)

            # map point from i-spacing * len(points_to_i) * 0.5 to i+spacing * len(points_to_i) * 0.5
            new_points = [(k, i - 0.3 + idx * spacing * 0.6) for idx, k in enumerate(points_to_i)]

            for k, v in new_points:
                file.write(f"{k},{v},{dataset_labels[k]}\n")

        # for k, v in embedding.items():
        #     file.write(f"{k},{v},{dataset_labels[k]}\n")


def train(dataset_features, dataset_labels, dataset_name, using=USING):
    best_embedding = OrderedDict()
    min_cost = float("inf")

    if USE_MULTIPROCESS:
        cpu_count = multiprocessing.cpu_count()
    else:
        cpu_count = 1

    process_pool = Pool(cpu_count)

    num_points = len(dataset_labels)

    constraints = format_ml_dataset(dataset_features, dataset_labels, subsample_factor=0, using=using,
                                    dataset_name=dataset_name)

    EPSILON = 1 / len(set(dataset_labels))

    # Don't split in train and testing, learn with respect to all constraints
    process_pool_arguments = format_arguments(constraints, num_points, cpu_count, EPSILON)
    responses = process_pool.starmap(lloc, process_pool_arguments)
    # out = lloc(constraints, num_points, constraints, 0)
    best_embedding = reduce_embedding(best_embedding, min_cost, responses)

    create_dataset_from_embedding(best_embedding, dataset_labels, dataset_name, using=using)


if __name__ == "__main__":
    for x, y, dataset_name in load_datasets():
        for using in ["features", "labels"]:
            print(f"[{using.upper()}]Doing {dataset_name}")
            embedding = train(x, y, dataset_name, using=using)
    

    for x, y, dataset_name in load_poisoned():
        for using in ["features", "labels"]:
            print(f"[{using.upper()}]Doing {dataset_name}")
            embedding = train(x, y, dataset_name, using=using)

