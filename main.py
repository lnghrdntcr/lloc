import numpy as np
import networkx as nx
from itertools import combinations, permutations
import random
from format_dataset import get_ds
from IPython import embed
from tqdm import tqdm
from math import factorial
from os import mkdir, system
from PIL import Image
from io import BytesIO
import requests as req

EPSILON = 1 / 10


def save_results(embeddings, image_cache, crop_map):
    for image_index, directory in embeddings.items():
        url = image_cache[image_index]
        response = req.get(url.replace("\"", ""))
        if response.status_code == 200:
            img_file = Image.open(BytesIO(response.content))
            h, w = img_file.size
            tlc, brc, tlr, brr = crop_map[image_index]

            top_left_col = h * float(tlc)
            bottom_right_col = h * float(brc)
            top_left_row = w * float(tlr)
            bottom_right_row = w * float(brr)
            face = img_file.crop((top_left_col, top_left_row, bottom_right_col, bottom_right_row))
            face.save(f"./results/{directory}/{image_index}.jpg")


def n_choose_k(n, k):
    assert n > k
    num = factorial(n)
    den = factorial(k) * factorial(n - k)

    return num / den


def setup_results_directories():
    system("rm -rf results/*")
    for i in range(int(1 / EPSILON) + 1):
        mkdir(f"./results/{i}")


def feedback_arc_set(G: nx.DiGraph, reorient=True):
    """
    :param G: Input Directed graph
    :param reorient Reorient the edges instead of returning the FAS
    :return: The feedback arc set of G, if reorient is set, it reorients the edges of the FAS
    """
    ret = G.copy()
    cycles = list(nx.simple_cycles(ret))

    for cycle in cycles:
        cycle_generating_vertices = list(combinations(cycle, 2))
        for vertices in cycle_generating_vertices:
            u, v = vertices
            try:
                ret.remove_edge(u, v)
                ret.add_edge(v, u)
                # cur_cycles = list(nx.simple_cycles(ret))
                # if not cur_cycles:
                #     return ret.copy()
            except nx.exception.NetworkXError:
                pass

    return ret.copy()


def get_buckets(arr, num_buckets):
    buckets = []
    bucketing_factor = int(len(arr) // num_buckets)
    last_base = (num_buckets - 1) * bucketing_factor
    for idx in range(num_buckets - 1):
        base = idx * bucketing_factor
        buckets.append(arr[base: base + bucketing_factor])
    buckets.append(arr[last_base:])
    return buckets


def format_embedding(base, embedding, mapped_to_representatives):
    ret = {
        str(base): 0
    }

    for i, el in enumerate(embedding):
        ret[str(el)] = i + 1

    for key, value in mapped_to_representatives.items():
        ret[key] = value

    return ret


def pretty_print_embedding(embedding):
    pretty_print = ", ".join([f"{el} -> {idx + 1}" for idx, el in enumerate(embedding)])
    print(f"Current embedding into R^1 = [{i} -> 0, {pretty_print}]")


def count_violated_constraints(embedding, constraints):
    count = 0

    for constraint in constraints:
        unrolled_constraints = combinations(constraint, 2)
        for unrolled_constraint in unrolled_constraints:
            assert len(unrolled_constraint) == 2
            s, t = unrolled_constraint

            if (f_s := embedding.get(str(s))) and (f_t := embedding.get(str(t))):
                count += int(f_s > f_t)
            else:
                count += 1

    return count


if __name__ == '__main__':
    setup_results_directories()

    idx_constraints, reverse_cache, crop_map = get_ds("./datasets/FEC_dataset/faceexp-comparison-data-test-public.csv", "test", early_stop_count=200)
    points = len(reverse_cache)

    best_embedding = {}
    best_violated_constraints = float("inf")
    terminate = False
    for i in tqdm(range(points), leave=False, desc="Points"):
        # find constraints where p_i is the first
        constraints = list(filter(lambda x: x[0] == i, idx_constraints))

        G = nx.DiGraph()
        # get the set of edges
        edges = list(map(lambda x: (x[1], x[2]), constraints))
        nodes = set([item for sublist in constraints for item in sublist if item != i])
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        reoriented = feedback_arc_set(G)

        try:
            topological_ordered_nodes = list(nx.topological_sort(reoriented))
        except Exception:
            continue
        num_buckets = int(1 / EPSILON)
        buckets = get_buckets(topological_ordered_nodes, num_buckets)
        if not buckets[0]:
            continue
        representatives = []
        representatives_map = {}
        for idx, bucket in enumerate(buckets):
            representatives.append(random.choice(bucket))

            for el in bucket:
                if el != representatives[idx]:
                    representatives_map[str(el)] = idx + 1

        # TODO: find better name for representative map
        all_embeddings = permutations(representatives, len(representatives))
        for raw_embedding in tqdm(all_embeddings, leave=False, desc="Embeddings"):
            embedding = format_embedding(i, raw_embedding, representatives_map)
            n_violated_constraints = count_violated_constraints(embedding, idx_constraints)

            if n_violated_constraints < best_violated_constraints:
                best_embedding = embedding
                best_violated_constraints = n_violated_constraints

    print(f"Best embedding with {best_violated_constraints} errors over {int(n_choose_k(len(idx_constraints), 2))} constraints -> {best_embedding}")

    save_results(best_embedding, reverse_cache, crop_map)
    embed()
