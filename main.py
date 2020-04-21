import multiprocessing
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np

from config import SUPPORTED_DATASETS, USE_PAGERANK, MNIST_CONSTRAINT_INCLUSION_PROBABILITY, MNIST_ERROR_RATE
from format_dataset import format_mnist_from_labels
from llcc import llcc, pagerank_llcc
from utils import n_choose_k, setup_results_directories, save_mnist_image, format_arguments


def main(USE_PAGERANK=USE_PAGERANK):
    if not USE_PAGERANK:
        print("LLCC")
        llcc_fn = llcc
    else:
        llcc_fn = pagerank_llcc

    best_embedding = OrderedDict()
    best_violated_constraints = float("inf")
    cpu_count = multiprocessing.cpu_count()
    process_pool = Pool(cpu_count)

    idx_constraints, reverse_cache, (x, y), class_distribution = format_mnist_from_labels(
        inclusion_probability=MNIST_CONSTRAINT_INCLUSION_PROBABILITY, error_probability=MNIST_ERROR_RATE)
    max_freq = np.max(class_distribution)
    print(
        f"Class distributions -> {', '.join([str(i) + ' -> ' + str(max_freq / el) for i, el in enumerate(class_distribution)])}")
    num_points = len(reverse_cache)
    process_pool_arguments = format_arguments(idx_constraints, num_points, multiprocessing.cpu_count(),
                                              class_distribution,
                                              use_pagerank=USE_PAGERANK)

    responses = process_pool.starmap(llcc_fn, process_pool_arguments)
    for embedding, n_violated_constraints in responses:
        if n_violated_constraints < best_violated_constraints:
            best_embedding = embedding
            best_violated_constraints = n_violated_constraints

    print(
        f"Best embedding with {best_violated_constraints} errors over {3 * len(idx_constraints)} constraints. Max possible constraints -> {num_points * int(n_choose_k(num_points, 2))} ")

    # for counter, (key, embed_to) in enumerate(best_embedding.items()):
    #     index = int(key)
    #     save_mnist_image(x[index], embed_to, index, bucketing=not USE_PAGERANK, image_name=counter)

    sorted_embedding = [int(k) for k, _ in sorted(best_embedding.items(), key=lambda x: x[1])]
    for i, el in enumerate(sorted_embedding):
        index = str(el)
        photo = x[el]
        save_mnist_image(photo, reverse_cache[index], 0, image_name=i)

    process_pool.close()

    print("DONE! Exiting...")
    exit(0)


if __name__ == "__main__":
    # Build dir structure for the results
    for ds in SUPPORTED_DATASETS:
        setup_results_directories(ds)

    main()
