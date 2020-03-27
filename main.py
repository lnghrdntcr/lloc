from format_dataset import fast_format_mnist
from IPython import embed
from multiprocessing import Pool
import multiprocessing
from llcc import llcc
from utils import n_choose_k, setup_results_directories, save_mnist_image, split_dataset

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
    #     "./datasets/FEC_dataset/faceexp-comparison-data-test-public.csv", early_stop_count=300)

    idx_constraints, reverse_cache, (x, y) = fast_format_mnist()
    num_points = len(reverse_cache)

    process_pool_arguments = split_dataset(idx_constraints, num_points, multiprocessing.cpu_count())


    #best_embedding, best_violated_constraints = llcc(num_points, idx_constraints, all_dataset, thread_id)

    responses = process_pool.starmap(llcc, process_pool_arguments)

    for embedding, n_violated_constraints in responses:
        if n_violated_constraints < best_violated_constraints:
            best_embedding = embedding
            best_violated_constraints = n_violated_constraints

    print(
        f"Best embedding with {best_violated_constraints} errors over {int(n_choose_k(len(idx_constraints), 2))} constraints -> {best_embedding}")

    for key, value in reverse_cache.items():
        try:
            embedding_key = str(best_embedding[str(key)])
        except KeyError:
            continue
        index = int(key)
        save_mnist_image(x[index], embedding_key, index)

    embed()
