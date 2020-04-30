import multiprocessing
from collections import OrderedDict
from itertools import combinations
from multiprocessing import Pool

from tqdm import tqdm

from config import SUPPORTED_DATASETS, USE_PAGERANK, MNIST_ERROR_RATE, \
    MNIST_DIGIT_EXCLUSION_PROBABILITY, EPSILON, USE_MULTIPROCESS
from format_dataset import format_mnist_from_labels
from llcc import llcc, pagerank_llcc
from utils import setup_results_directories, format_arguments, \
    train_test_split
from IPython import embed

def main(USE_PAGERANK=USE_PAGERANK):
    if not USE_PAGERANK:
        print("LLCC")
        llcc_fn = llcc
    else:
        llcc_fn = pagerank_llcc

    best_embedding = OrderedDict()
    min_cost = float("inf")

    if USE_MULTIPROCESS:
        cpu_count = multiprocessing.cpu_count()
    else:
        cpu_count = 1

    process_pool = Pool(cpu_count)

    idx_constraints, reverse_cache, (x,
                                     y), class_distribution = format_mnist_from_labels()  # format_google_ds("./datasets/FEC_dataset/faceexp-comparison-data-train-public.csv", smart_constraints=False, early_stop_count=10000)

    num_points = len(reverse_cache)
    train_constraints, test_constraints = train_test_split(idx_constraints, test_percentage=0.2)

    process_pool_arguments = format_arguments(train_constraints, num_points, cpu_count,
                                              use_pagerank=USE_PAGERANK)
    responses = process_pool.starmap(llcc_fn, process_pool_arguments)

    for embedding, n_violated_constraints in responses:
        if n_violated_constraints < min_cost:
            best_embedding = embedding
            min_cost = n_violated_constraints

    # FIXME: This line outputs an accuracy much worse than what's real because min_cost is now a cost and not a
    # count of violated constraints
    # Still indicative tho
    print(
        f"Best embedding with {min_cost} errors over {3 * len(idx_constraints)} constraints. Precision {100 - min_cost / (3 * len(idx_constraints)) * 100}%.")
    process_pool.close()
    embed()
    error_rate = 0
    missing = 0
    for test_constraint in tqdm(test_constraints, desc="Testing..."):
        train_constraints.append(test_constraint)
        new_cost = 0
        for i, j in list(combinations(test_constraint, 2))[:-1]:
            if ((f_i := best_embedding.get(str(i))) is not None) and ((f_j := best_embedding.get(str(j))) is not None):
                new_cost += int(f_i > f_j)
            else:
                missing += 1
                new_cost += 1

        if new_cost != 0:
            error_rate += 1

        del train_constraints[-1]

    print(f"Error Percentage: {(error_rate / len(test_constraints)) * 100}%")
    print(f"Missing {missing}")

    print("DONE! Exiting...")
    exit(0)


if __name__ == "__main__":
    print(
        f"Running test with EPSILON={EPSILON}, MNIST_ERROR_RATE={MNIST_ERROR_RATE}, MNIST_DIGIT_EXCLUSION_PROBABILITY={MNIST_DIGIT_EXCLUSION_PROBABILITY}")

    # Build dir structure for the results
    for ds in SUPPORTED_DATASETS:
        setup_results_directories(ds)

    main()
