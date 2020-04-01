from format_dataset import format_mnist_from_labels, format_google_ds, format_mnist_from_correlations
from IPython import embed
from multiprocessing import Pool
import multiprocessing
from llcc import llcc
from utils import n_choose_k, setup_results_directories, save_mnist_image, format_arguments, save_fec_results

from config import SUPPORTED_DATASETS

if __name__ == "__main__":

    best_embedding = {}
    best_violated_constraints = float("inf")
    cpu_count = multiprocessing.cpu_count()
    process_pool = Pool(cpu_count)

    # Build dir structure for the results
    for ds in SUPPORTED_DATASETS:
        setup_results_directories(ds)

    # idx_constraints, reverse_cache, crop_map = format_google_ds(
    #      "./datasets/FEC_dataset/faceexp-comparison-data-test-public.csv", early_stop_count=2000)

    # format_mnist_from_correlations()
    idx_constraints, reverse_cache, (x, y) = format_mnist_from_correlations()
    num_points = len(reverse_cache)

    process_pool_arguments = format_arguments(idx_constraints, num_points, multiprocessing.cpu_count())

    responses = process_pool.starmap(llcc, process_pool_arguments)

    for embedding, n_violated_constraints in responses:
        if n_violated_constraints < best_violated_constraints:
            best_embedding = embedding
            best_violated_constraints = n_violated_constraints

    # constraints -> {best_embedding}
    print(
        f"Best embedding with {best_violated_constraints} errors over {3 * len(idx_constraints)} constraints. Max possible constraints -> {num_points * int(n_choose_k(num_points, 2))} ")

    # save_fec_results(best_embedding, reverse_cache, crop_map)

    for key, value in reverse_cache.items():
        try:
            embedding_key = str(best_embedding[str(key)])
        except KeyError:
            continue
        index = int(key)
        save_mnist_image(x[index], embedding_key, index, correlation=True)

    process_pool.close()
    print("DONE! Exiting...")
    exit(0)
