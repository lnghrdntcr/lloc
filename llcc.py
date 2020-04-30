import random
from collections import OrderedDict
from itertools import combinations

import networkx as nx
import numpy as np
from tqdm import tqdm

from config import EPSILON, USE_DISTANCE


def feedback_arc_set(G: nx.DiGraph, process_id=0):
    """
    Constructs a DAG by removing edges of the feedback arc set incrementally
    :param G: Input graph
    :param process_id: ID of the process, in case of multiprocessing enabled
    :return: A DAG
    """
    ret = G.copy()
    for node in tqdm(ret.nodes, position=process_id * 2 + 1, leave=False, desc=f"[Core {process_id}] FAS        "):
        in_edges = ret.in_edges(node)
        neighbouring_nodes = list(map(lambda x: x[0] if x[1] == node else x[1], in_edges))

        for neighbor in neighbouring_nodes:
            try:
                next(nx.simple_cycles(ret))
                path_from = nx.shortest_path(ret, source=node, target=neighbor)
                random_idx = random.randint(0, len(path_from) - 1)
                ret.remove_edge(path_from[random_idx], path_from[(random_idx + 1) % len(path_from)])

                path_to = nx.shortest_path(ret, source=neighbor, target=node)
                random_idx = random.randint(0, len(path_to) - 1)
                ret.remove_edge(path_to[random_idx], path_to[(random_idx + 1) % len(path_to)])
            except (nx.NetworkXNoPath, nx.NetworkXError):
                continue
            except StopIteration:
                return ret

    return ret


def get_buckets(arr, num_buckets, bucketed_class_distribution=None):
    """
    Creates the buckets with respect to a given class distribution
    :param arr: Elements to be partitioned in subsets
    :param num_buckets: Number of subsets
    :param bucketed_class_distribution: Distribution of elements in the subsets
    :return:
    """
    buckets = []

    if bucketed_class_distribution is None:
        bucketed_class_distribution = [1 / num_buckets for _ in range(num_buckets)]

    base_idx = 0
    for elements_distribution in bucketed_class_distribution[:-1]:
        until_idx = base_idx + int(elements_distribution * len(arr))
        buckets.append(arr[base_idx: until_idx])
        base_idx = until_idx

    buckets.append(arr[base_idx:])
    return buckets


def format_embedding(base, representatives, mapped_to_representatives, move_pattern=None):
    """
    Creates the initial embedding in which all elements are mapped to their representative in the bucket
    and the representatives are mapped to their index in the array
    :param base: The point to be mapped to 0
    :param representatives: The representative points
    :param mapped_to_representatives: Points in the buckets
    :param move_pattern: Swap representatives
    :return: An embedding
    """
    ret = OrderedDict()
    ret[str(base)] = 0

    for i, el in enumerate(representatives):
        ret[str(el)] = i + 1

    for key, maps_to in list(ret.items()):
        # get all_elements that are mapped to `maps_to`
        bucket_elements = list(filter(lambda x: x[1] == maps_to, mapped_to_representatives.items()))
        for key, _ in bucket_elements:

            if move_pattern is not None:
                if maps_to == move_pattern[0]:
                    new_maps_to = move_pattern[1]
                elif maps_to == move_pattern[1]:
                    new_maps_to = move_pattern[0]
                else:
                    new_maps_to = maps_to
            else:
                new_maps_to = maps_to

            position = new_maps_to
            ret[str(key)] = position

    return ret


def count_violated_constraints(embedding, constraints, representatives, constraints_weights, ignore_missing_values=False,
                               use_distance=USE_DISTANCE):
    count = 0
    for idx, constraint in enumerate(constraints):
        if use_distance:
            s, t, k = constraint
            f_i = embedding.get(str(representatives[s - 1]))
            f_j = embedding.get(str(representatives[t - 1]))
            f_k = embedding.get(str(representatives[k - 1]))
            if (f_i is not None) and (f_j is not None) and (f_k is not None):
                count += int(np.abs(f_i - f_j) >= np.abs(f_i - f_k)) * constraints_weights[idx]
            elif not ignore_missing_values:
                count += 1
        else:
            constraints_combinations = combinations(constraints, 2)

            for c in constraints_combinations:
                s, t = c
                f_i = embedding.get(str(representatives[s - 1]))
                f_j = embedding.get(str(representatives[t - 1]))
                if (f_i is not None) and (f_j is not None):
                    count += int(f_i >= f_j) * constraints_weights[idx]
                elif not ignore_missing_values:
                    count += 1
    return count


def build_graph_from_constraints(constraints, cur_vertex):
    """
    Builds the tournament from the given constraints
    :param cur_vertex: The vertex to consider as first point
    :param constraints: Triplet Constraints
    :return: The built turnament
    """
    G = nx.DiGraph()
    edges = list(map(lambda x: (x[1], x[2]), constraints))
    nodes = set([item for sublist in constraints for item in sublist if item != cur_vertex])
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def patch_embedding_from_all_constraints(representative_map, representatives, constraints):
    """
    Maps every point that's missing in the embedding to 0
    :param representative_map: 
    :param representatives: 
    :param constraints: 
    :return: 
    """
    ret = representative_map.copy()
    for constraint in constraints:
        assert len(constraint) == 3
        for point in constraint:
            if not representative_map.get(str(point)):
                if point not in representatives:
                    # Map to bucket 0 every element missing from the map
                    ret[str(point)] = 0
                else:
                    ret[str(point)] = representatives.index(point) + 1

    return ret


def build_representative_embedding(buckets):
    representatives = []
    embedding = OrderedDict()

    for idx, bucket in enumerate(buckets):
        representatives.append(random.choice(bucket))

        for el in bucket:
            if el != representatives[idx]:
                embedding[str(el)] = idx + 1

    return representatives, embedding


def search_better_embedding(all_dataset, current_best_embedding, current_best_violated_constraints, point_id,
                            representatives, representatives_map, constraints_weight, process_id, num_points = 0):
    local_best_embedding = None
    local_best_violated_constraints = float("inf")

    for i in tqdm(range(int(1 / EPSILON) - 1), position=process_id * 2 + 1, leave=False,
                  desc=f"[Core {process_id}] Embeddings "):
        next_representatives = representatives.copy()
        next_representatives[i], next_representatives[i + 1] = next_representatives[i + 1], next_representatives[i]

        tmp_representative_map = dict([(k, v) for k, v in representatives_map.items() if int(k) not in representatives])
        if local_best_embedding is None:
            local_best_embedding = format_embedding(point_id, representatives, tmp_representative_map)

        next_embedding = format_embedding(point_id, next_representatives, tmp_representative_map, move_pattern=(i, i + 1))

        if local_best_violated_constraints == float("inf"):
            local_best_violated_constraints = count_violated_constraints(local_best_embedding, all_dataset, representatives,
                                                                         constraints_weight)

        next_violated_constraints = count_violated_constraints(next_embedding, all_dataset, next_representatives, constraints_weight)

        if local_best_violated_constraints > next_violated_constraints:
            representatives = next_representatives
            representatives_map = dict([(k, v) for k, v in next_embedding.items() if int(k) not in representatives])
            local_best_embedding = next_embedding.copy()
            local_best_violated_constraints = next_violated_constraints

    embedding = local_best_embedding.copy()
    violated_constraints = local_best_violated_constraints

    if violated_constraints < current_best_violated_constraints:
        current_best_embedding = embedding
        current_best_violated_constraints = violated_constraints

    return current_best_embedding, representatives, current_best_violated_constraints


def create_wlcc(representatives_map, all_dataset, use_distance=USE_DISTANCE):
    new_constraints = dict()
    weights = []
    for constraint in all_dataset:
        i, j, k = constraint
        mapped_i, mapped_j, mapped_k = representatives_map[str(i)], representatives_map[str(j)], representatives_map[
            str(k)]
        if new_constraints.get((mapped_i, mapped_j, mapped_k)) is None:
            new_constraints[(mapped_i, mapped_j, mapped_k)] = 1
        else:
            new_constraints[(mapped_i, mapped_j, mapped_k)] += 1

    for idx, (constraint, count) in enumerate(new_constraints.items()):
        if use_distance:
            constraint_combinations = combinations(constraint, 2)
            acc = 0
            for c in constraint_combinations:
                i, j = c
                if representatives_map[str(i)] >= representatives_map[str(j)]:
                    acc += 1

            weights.append(count * acc)

        else:
            i, j, j = constraint
            if np.abs(representatives_map[str(i)] - representatives_map[str(j)]) >= np.abs(
                    representatives_map[str(i)] - representatives_map[str(k)]):
                weights.append(count)
            else:
                weights.append(0)

    new_idx_constraints = [constraint for constraint, count in new_constraints.items()]
    return representatives_map, new_idx_constraints, weights


def llcc(idx_constraints, num_points, all_dataset, process_id):
    """
    Learns a line from the given ordinal constraints
    """
    best_embedding = None
    best_violated_constraints = float("inf")
    num_buckets = int(1 / EPSILON)
    for i in tqdm(range(num_points), position=process_id * 2, leave=False, desc=f"[Core {process_id}] Points     "):
        # find constraints where p_i is the first
        point_id = i + process_id * num_points
        constraints = list(filter(lambda x: x[0] == point_id, idx_constraints))
        G = build_graph_from_constraints(constraints, point_id)
        reoriented = feedback_arc_set(G, process_id=process_id)
        try:
            topological_ordered_nodes = list(nx.topological_sort(reoriented))
        except nx.exception.NetworkXUnfeasible:
            # Skip iteration if a topological ordering cannot be built
            continue

        buckets = get_buckets(topological_ordered_nodes, num_buckets)

        if not buckets[0]:
            # Skip iteration if the point had no constraints to begin with
            continue

        # From here I have to build the WLCC instance
        representatives, base_embedding = build_representative_embedding(buckets)
        base_embedding = patch_embedding_from_all_constraints(base_embedding, representatives,
                                                              all_dataset)

        base_embedding, projected_constraints, constraints_weights = create_wlcc(base_embedding, all_dataset)

        embedding, new_representatives, _ = search_better_embedding(projected_constraints, best_embedding,
                                                                  best_violated_constraints, point_id,
                                                                  representatives, base_embedding,
                                                                  constraints_weights, process_id, num_points=num_points)

        embedding = patch_embedding_from_all_constraints(embedding, new_representatives, all_dataset)
        violated_constraints = count_violated_constraints(embedding, projected_constraints, new_representatives, constraints_weights)
        if violated_constraints < best_violated_constraints:
            best_embedding = embedding.copy()
            best_violated_constraints = violated_constraints

    return best_embedding, best_violated_constraints

