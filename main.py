import networkx as nx
from itertools import permutations
from format_dataset import format_google_ds, fast_format_mnist
from IPython import embed
from tqdm import tqdm

from llcc import feedback_arc_set, get_buckets, format_embedding, count_violated_constraints, \
    build_graph_from_constraints, build_representative_embedding, patch_representative_map_from_all_constraints
from utils import save_results, n_choose_k, setup_results_directories, save_mnist_image

from config import EPSILON, SUPPORTED_DATASETS, MNIST_SUBSAMPLE_FACTOR


if __name__ == "__main__":

    # Build dir structure for the results
    for ds in SUPPORTED_DATASETS:
        setup_results_directories(ds)

    # idx_constraints, reverse_cache, crop_map = format_google_ds(
    #     "./datasets/FEC_dataset/faceexp-comparison-data-test-public.csv", early_stop_count=300)

    idx_constraints, reverse_cache, (x, y) = fast_format_mnist()
    points = len(reverse_cache)

    best_embedding = {}
    best_violated_constraints = float("inf")
    terminate = False

    for i in tqdm(range(points), leave=False, desc="Points    "):
        # find constraints where p_i is the first
        constraints = list(filter(lambda x: x[0] == i, idx_constraints))

        G = build_graph_from_constraints(constraints, i)

        reoriented = feedback_arc_set(G)

        try:
            topological_ordered_nodes = list(nx.topological_sort(reoriented))
        except nx.exception.NetworkXUnfeasible:
            # Skip iteration if a topological ordering cannot be built
            continue

        num_buckets = int(1 / EPSILON)
        buckets = get_buckets(topological_ordered_nodes, num_buckets)

        if not buckets[0]:
            # Skip iteration if the point had no constraints to begin with
            continue

        representatives, representatives_map = build_representative_embedding(buckets)

        representatives_map = patch_representative_map_from_all_constraints(representatives_map, idx_constraints)

        # Perform exhaustive enumeration among all representatives
        all_embeddings = permutations(representatives, len(representatives))

        for raw_embedding in tqdm(list(all_embeddings), leave=False, desc="Embeddings"):
            embedding = format_embedding(i, raw_embedding, representatives_map)
            n_violated_constraints = count_violated_constraints(embedding, idx_constraints)

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
