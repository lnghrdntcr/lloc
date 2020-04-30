import multiprocessing
from collections import OrderedDict
from multiprocessing import Pool

from config import SUPPORTED_DATASETS, USE_PAGERANK, MNIST_ERROR_RATE, \
    MNIST_DIGIT_EXCLUSION_PROBABILITY, EPSILON, USE_MULTIPROCESS, USE_DISTANCE
from format_dataset import format_mnist_from_labels
from llcc import llcc, pagerank_llcc
from utils import n_choose_k, setup_results_directories, save_mnist_image, format_arguments, \
    select_bucket_from_embedding_value


def main(USE_PAGERANK=USE_PAGERANK):
    if not USE_PAGERANK:
        print("LLCC")
        llcc_fn = llcc
    else:
        llcc_fn = pagerank_llcc

    best_embedding = OrderedDict()
    best_violated_constraints = float("inf")

    if USE_MULTIPROCESS:
        cpu_count = multiprocessing.cpu_count()
    else:
        cpu_count = 1

    process_pool = Pool(cpu_count)

    idx_constraints, reverse_cache, (x, y), class_distribution = format_mnist_from_labels()  # format_google_ds("./datasets/FEC_dataset/faceexp-comparison-data-train-public.csv", smart_constraints=False, early_stop_count=10000)

    num_points = len(reverse_cache)

    # bucket_distribution = map_class_distribution_to_bucket_distribution(class_distribution)

    process_pool_arguments = format_arguments(idx_constraints, num_points, cpu_count,
                                              use_pagerank=USE_PAGERANK)
    responses = process_pool.starmap(llcc_fn, process_pool_arguments)

    for embedding, n_violated_constraints in responses:
        if n_violated_constraints < best_violated_constraints:
            best_embedding = embedding
            best_violated_constraints = n_violated_constraints

    print(
        f"Best embedding with {best_violated_constraints} errors over {3 * len(idx_constraints)} constraints. Precision {100 - best_violated_constraints / (3 * len(idx_constraints)) * 100}%. Max possible constraints -> {num_points * int(n_choose_k(num_points, 2))} ")

    # embed()
    # save_csv_results(EPSILON, best_violated_constraints, 3 * len(idx_constraints), "LLCC", "DIGIT_DISTANCE", MNIST_DIGIT_EXCLUSION_PROBABILITY, MNIST_ERROR_RATE)
    sorted_embedding = [(int(k), v) for k, v in sorted(best_embedding.items(), key=lambda x: x[1])]
    for i, (el, value) in enumerate(sorted_embedding):
        index = str(el)
        photo = x[el]

        bucket = select_bucket_from_embedding_value(value, class_distribution)

        save_mnist_image(photo, index, 0, image_name=i, bucketing=False)

    process_pool.close()

    print("DONE! Exiting...")
    exit(0)


if __name__ == "__main__":
    print(
        f"Running test with EPSILON={EPSILON}, MNIST_ERROR_RATE={MNIST_ERROR_RATE}, MNIST_DIGIT_EXCLUSION_PROBABILITY={MNIST_DIGIT_EXCLUSION_PROBABILITY}")

    # Build dir structure for the results
    for ds in SUPPORTED_DATASETS:
        setup_results_directories(ds)

    main()
