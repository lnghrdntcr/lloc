import multiprocessing
import sys
from collections import OrderedDict
from multiprocessing import Pool

from tqdm import tqdm

import utils.read_dataset as rd
from format_datasets.format_dataset import format_ml_dataset
from lloc import lloc, create_nd_embedding, get_violated_constraints, count_raw_violated_constraints
from utils.config import USING, USE_MULTIPROCESS, EPSILON, SECOND_DIM
from utils.utils import format_arguments, \
    get_num_points, reduce_embedding, merge_embeddings


def create_dataset_from_embedding(embedding, dataset_labels, dataset_name, using="features", n_dims=1):
    filename = f"./datasets/lloc_output/{dataset_name}-from-{using}-{n_dims}d.csv"
    with open(filename, "w+") as file:
        # write header

        if n_dims == 1:
            file.write("idx,x,label\n")
            for i in tqdm(range(int(1 / EPSILON)), desc=f"[{dataset_name.upper()}] Writing new embedding"):

                points_to_i = [k for k, v in embedding.items() if v == i]
                if len(points_to_i) == 0:
                    continue
                spacing = 1 / len(points_to_i)

                # map point from i-spacing * len(points_to_i) * 0.5 to i+spacing * len(points_to_i) * 0.5
                new_points = [(k, i - 0.3 + idx * spacing * 0.6) for idx, k in enumerate(points_to_i)]

                for k, v in new_points:
                    file.write(f"{k},{v},{dataset_labels[k]}\n")

        else:
            file.write("idx,x,y,label\n")

            for idx, (v1, v2) in tqdm(embedding.items(), desc=f"[{dataset_name.upper()}] Writing new embedding"):
                file.write(f"{idx},{v1},{v2},{dataset_labels[idx]}\n")



def train(dataset_features, dataset_labels, dataset_name, using=USING):
    best_embedding = OrderedDict()
    min_cost = float("inf")
    n_dims = 1
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
    create_dataset_from_embedding(best_embedding, dataset_labels, dataset_name, using=using, n_dims=1)

    if SECOND_DIM:
        n_dims = 2
        new_train_set, _ = get_violated_constraints(best_embedding, constraints)

        new_num_points = get_num_points(new_train_set)

        process_pool_args = format_arguments(new_train_set, new_num_points, cpu_count, EPSILON)
        responses = process_pool.starmap(lloc, process_pool_args)
        new_best_embedding = reduce_embedding(OrderedDict(), float("inf"), responses)

        projected_best_embedding = create_nd_embedding(best_embedding, n_dim=2)
        best_violation_count = count_raw_violated_constraints(projected_best_embedding, constraints)
        new_embedding = projected_best_embedding.copy()
        new_violation_count = best_violation_count
        print(f"Original Violates {new_violation_count / len(constraints) * 100}% constraints", file=sys.stderr)
        new_violation_count, best_embedding = merge_embeddings(new_best_embedding, new_violation_count,
                                                               new_embedding, constraints)
        print(best_embedding)
        print(f"New Violates {new_violation_count/ len(constraints) * 100}% constraints", file=sys.stderr)
        process_pool.close()

        create_dataset_from_embedding(best_embedding, dataset_labels, dataset_name, using=using, n_dims=n_dims)


if __name__ == "__main__":
    for x, y, dataset_name in rd.load_datasets():
        for using in ["labels", "features"]:
            print(f"[{using.upper()}]Doing {dataset_name}")
            train(x, y, dataset_name, using=using)

    for x, y, dataset_name in rd.load_poisoned():
        for using in ["labels", "features"]:
            print(f"[{using.upper()}]Doing {dataset_name}")
            train(x, y, dataset_name, using=using)

