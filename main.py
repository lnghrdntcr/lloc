from format_dataset import format_mnist_from_labels, format_google_ds, format_mnist_from_correlations
from IPython import embed
from multiprocessing import Pool
import multiprocessing
from llcc import llcc, pagerank_llcc, reorient_cycle_generating_edges
from utils import n_choose_k, setup_results_directories, save_mnist_image, format_arguments, save_fec_results
from config import SUPPORTED_DATASETS, USE_PAGERANK


def main(USE_PAGERANK=False):

    if not USE_PAGERANK:
        print("LLCC")
        llcc_fn = llcc
    else:
        llcc_fn = pagerank_llcc
    best_embedding = {}
    best_violated_constraints = float("inf")
    cpu_count = multiprocessing.cpu_count()
    process_pool = Pool(cpu_count)

    # idx_constraints, reverse_cache, crop_map = format_google_ds(
    #     "./datasets/FEC_dataset/faceexp-comparison-data-test-public.csv", early_stop_count=600, smart_constraints=True)

    idx_constraints, reverse_cache, (x, y), error_count = format_mnist_from_labels()
    num_points = len(reverse_cache)
    process_pool_arguments = format_arguments(idx_constraints, num_points, multiprocessing.cpu_count(),
                                              use_pagerank=USE_PAGERANK)
    responses = process_pool.starmap(llcc_fn, process_pool_arguments)
    for embedding, n_violated_constraints in responses:
        if n_violated_constraints < best_violated_constraints:
            best_embedding = embedding
            best_violated_constraints = n_violated_constraints
    # constraints -> {best_embedding}
    print(
        f"Best embedding with {best_violated_constraints} errors over {3 * len(idx_constraints)} constraints. Max possible constraints -> {num_points * int(n_choose_k(num_points, 2))} ")
    # save_fec_results(best_embedding, reverse_cache, crop_map, directory=False)
    for key, value in reverse_cache.items():
        try:
            embedding_key = str(best_embedding[str(key)])
        except KeyError:
            continue
        index = int(key)
        save_mnist_image(x[index], embedding_key, index, bucketing=not USE_PAGERANK)
    process_pool.close()
    if not USE_PAGERANK:
        main(USE_PAGERANK=True)
    else:
        print("DONE! Exiting...")
        exit(0)

if __name__ == "__main__":
    # Build dir structure for the results
    for ds in SUPPORTED_DATASETS:
        setup_results_directories(ds)

    main()
