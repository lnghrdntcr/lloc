import networkx as nx
from itertools import permutations
from format_dataset import format_google_ds, format_mnist
from IPython import embed
from tqdm import tqdm

from llcc import feedback_arc_set, get_buckets, format_embedding, count_violated_constraints, \
    build_graph_from_constraints, build_representative_embedding
from utils import save_results, n_choose_k, setup_results_directories, save_mnist_image

from config import EPSILON, SUPPORTED_DATASETS, MNIST_SUBSAMPLE_FACTOR


if __name__ == "__main__":

    # Build dir structure for the results
    for ds in SUPPORTED_DATASETS:
        setup_results_directories(ds)

    # idx_constraints, reverse_cache, crop_map = format_google_ds(
    #     "./datasets/FEC_dataset/faceexp-comparison-data-test-public.csv", early_stop_count=300)

    idx_constraints, reverse_cache, (x, y) = format_mnist()
    points = len(reverse_cache)

    best_embedding = {}
    best_violated_constraints = float("inf")
    terminate = False

    for i in tqdm(range(points), leave=False, desc="Points"):
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

    #best_embedding = {'15': 0, '107': 1, '138': 2, '80': 3, '199': 1, '184': 1, '177': 1, '173': 1, '172': 1, '162': 1, '158': 1, '152': 1, '143': 1, '140': 1, '111': 1, '89': 1, '81': 1, '78': 1, '61': 1, '59': 1, '38': 1, '27': 1, '18': 1, '17': 1, '14': 1, '7': 1, '194': 1, '190': 1, '179': 1, '176': 1, '171': 1, '169': 1, '148': 1, '142': 1, '136': 1, '105': 1, '99': 1, '87': 1, '85': 1, '82': 1, '72': 1, '65': 1, '50': 1, '47': 1, '44': 1, '21': 1, '8': 1, '195': 1, '182': 1, '161': 1, '156': 1, '155': 1, '147': 1, '145': 1, '139': 1, '117': 1, '114': 1, '103': 1, '100': 1, '92': 1, '66': 1, '63': 1, '49': 1, '46': 1, '32': 1, '23': 2, '10': 2, '9': 2, '4': 2, '191': 2, '187': 2, '170': 2, '168': 2, '160': 2, '151': 2, '134': 2, '126': 2, '122': 2, '118': 2, '110': 2, '104': 2, '74': 2, '70': 2, '58': 2, '26': 2, '6': 2, '5': 2, '192': 2, '188': 2, '157': 2, '132': 2, '130': 2, '108': 2, '86': 2, '75': 2, '62': 2, '57': 2, '48': 2, '42': 2, '31': 2, '181': 2, '164': 2, '141': 2, '131': 2, '127': 2, '76': 2, '71': 2, '67': 2, '60': 2, '55': 2, '43': 2, '40': 2, '36': 2, '29': 2, '28': 2, '25': 2, '13': 2, '12': 2, '2': 2, '1': 2, '193': 2, '186': 2, '180': 2, '167': 2, '159': 2, '153': 2, '146': 3, '121': 3, '112': 3, '98': 3, '96': 3, '88': 3, '84': 3, '68': 3, '54': 3, '39': 3, '35': 3, '30': 3, '22': 3, '19': 3, '11': 3, '0': 3, '198': 3, '189': 3, '183': 3, '166': 3, '150': 3, '149': 3, '144': 3, '137': 3, '129': 3, '123': 3, '119': 3, '113': 3, '106': 3, '97': 3, '95': 3, '56': 3, '53': 3, '52': 3, '51': 3, '41': 3, '37': 3, '24': 3, '16': 3, '185': 3, '178': 3, '175': 3, '174': 3, '165': 3, '163': 3, '154': 3, '133': 3, '124': 3, '120': 3, '116': 3, '102': 3, '101': 3, '94': 3, '91': 3, '90': 3, '79': 3, '77': 3, '64': 3, '45': 3, '20': 3, '3': 3}

    for key, value in reverse_cache.items():
        try:
            embedding_key = str(best_embedding[str(key)])
        except KeyError:
            continue
        index = int(key)
        save_mnist_image(x[index], embedding_key, index)

    embed()
