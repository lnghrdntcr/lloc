import random
from collections import OrderedDict
from itertools import combinations

import networkx as nx
import numpy as np
from tqdm import tqdm

from config import EPSILON, USE_DISTANCE, BAR_POSITION_OFFSET


def feedback_arc_set(G: nx.DiGraph, process_id=0):
    """
    Constructs a DAG by removing edges of the feedback arc set incrementally
    :param G: Input graph
    :param process_id: ID of the process, in case of multiprocessing enabled
    :return: A DAG
    """
    ret = G.copy()
    for node in tqdm(ret.nodes, position=process_id * 2 + 1 + BAR_POSITION_OFFSET, leave=False,
                     desc=f"[Core {process_id}] FAS        "):
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


def create_buckets(arr, num_buckets, bucketed_class_distribution=None):
    """
    Creates the buckets with respect to a given class distribution
    :param arr: Elements to be partitioned in subsets
    :param num_buckets: Number of subsets
    :param bucketed_class_distribution: Distribution of elements in the subsets
    :return: all the points in the arr list divided into `num_buckets` buckets
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


def format_embedding(base_point, representatives, mapped_to_representatives, move_pattern=None):
    """
    Creates the initial embedding in which all elements are mapped to their representative in the bucket
    and the representatives are mapped to their index in the array
    :param base_point: The point to be mapped to 0
    :param representatives: The representative points
    :param mapped_to_representatives: Points in the buckets
    :param move_pattern: Swap representatives
    :return: An embedding
    """
    ret = OrderedDict()
    ret[base_point] = 0

    for i, el in enumerate(representatives):
        ret[el] = i + 1

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
            ret[key] = position

    return ret


def count_violated_constraints(embedding, constraints, representatives, constraints_weights,
                               ignore_missing_values=False,
                               use_distance=USE_DISTANCE):
    weight = 0
    for idx, constraint in enumerate(constraints):
        if use_distance:
            s, t, k = constraint
            f_i = embedding.get(representatives[s])
            f_j = embedding.get(representatives[t])
            f_k = embedding.get(representatives[k])
            if (f_i is not None) and (f_j is not None) and (f_k is not None):
                weight += int(np.abs(f_i - f_j) >= np.abs(f_i - f_k)) * constraints_weights[idx]
            elif not ignore_missing_values:
                weight += 1
        else:
            constraints_combinations = combinations(constraint, 2)

            for c in constraints_combinations:
                s, t = c
                f_i = embedding.get(representatives[s])
                f_j = embedding.get(representatives[t])
                if (f_i is not None) and (f_j is not None):
                    weight += int(f_i >= f_j) * constraints_weights[idx]
                elif not ignore_missing_values:
                    weight += 1
    return weight


def build_graph_from_constraints(constraints, cur_vertex):
    """
    Builds the tournament from the given constraints
    :param cur_vertex: The vertex to consider as first point
    :param constraints: Triplet Constraints
    :return: The built tournament
    """
    G = nx.DiGraph()
    edges = list(map(lambda x: (x[1], x[2]), constraints))
    nodes = set([item for sublist in constraints for item in sublist if item != cur_vertex])
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def patch_embedding_from_all_constraints(base_embedding, representatives, constraints):
    """
    Maps every point that's missing in the embedding to 0
    :param base_embedding: 
    :param representatives: 
    :param constraints: 
    :return: An embedding that contains all the points
    """
    ret = base_embedding.copy()
    for constraint in constraints:
        for point in constraint:
            if not base_embedding.get(point):
                if point not in representatives:
                    # Map to bucket 0 every element missing from the map
                    ret[point] = 0

    return ret


def build_representative_embedding(base_point, buckets):
    """
    Chooses the representatives for each bucket and maps every point in that bucket to the index of that bucket + 1
    :param base_point: The point that gets mapped to bucket 0
    :param buckets:
    :return: An embedding
    """
    representatives = [base_point]
    embedding = OrderedDict([(base_point, 0)])

    for idx, bucket in enumerate(buckets):
        representatives.append(random.choice(bucket))

        for el in bucket:
            # Since `base_point` is already mapped to 0
            if el != base_point:
                embedding[el] = idx + 1

    return representatives, embedding


def rename_constraints(constraints, first_token, second_token):
    new_constraints = []

    for i, constraint in enumerate(constraints):
        new_constraint = [*constraint]
        for j, point in enumerate(new_constraint):
            if point == first_token:
                new_constraint[j] = second_token
            elif point == second_token:
                new_constraint[j] = first_token
        new_constraints.append(new_constraint)

    return new_constraints


def search_better_embedding(dataset, best_embedding, best_weight_violated_constraints, base_point,
                            representatives, base_embedding, base_weight_violated_constraints, constraints_weight,
                            process_id=0):
    # Create reference embedding
    local_best_embedding = base_embedding
    local_best_weight = base_weight_violated_constraints
    local_constraints = dataset

    for i in tqdm(range(int(1 / EPSILON) - 1), position=process_id * 2 + 1 + BAR_POSITION_OFFSET, leave=False,
                  desc=f"[Core {process_id}] Embeddings "):
        # Swap contiguous pairs of representatives
        next_representatives = representatives.copy()
        next_representatives[i], next_representatives[i + 1] = next_representatives[i + 1], next_representatives[i]

        # Create the next embedding from the swapped representatives and the previous best embedding
        next_embedding = format_embedding(base_point, next_representatives, local_best_embedding,
                                          move_pattern=(i, i + 1))

        next_constraints = rename_constraints(local_constraints, i, i + 1)
        next_weight = count_violated_constraints(next_embedding, next_constraints, next_representatives,
                                                 constraints_weight)

        if local_best_weight > next_weight:
            representatives = next_representatives.copy()
            local_best_embedding = next_embedding
            local_best_weight = next_weight
            local_constraints = next_constraints

    if best_weight_violated_constraints > local_best_weight:
        return local_best_embedding, representatives, local_best_weight

    return best_embedding, representatives, best_weight_violated_constraints


def create_wlcc(embedding, dataset, use_distance=USE_DISTANCE):
    """
    Creates a WLCC instance
    :param embedding: 
    :param dataset: 
    :param use_distance: 
    :return: 
    """
    new_constraints = dict()
    weights = []
    for constraint in dataset:
        i, j, k = constraint
        mapped_i, mapped_j, mapped_k = embedding[i], embedding[j], embedding[k]

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
                if embedding[i] >= embedding[j]:
                    acc += 1

            weights.append(count * acc)

        else:
            i, j, k = constraint
            if np.abs(embedding[i] - embedding[j]) >= np.abs(
                    embedding[i] - embedding[k]):
                weights.append(count)
            else:
                weights.append(0)

    new_idx_constraints = [constraint for constraint, count in new_constraints.items()]
    return embedding, new_idx_constraints, weights


def llcc(idx_constraints, num_points, all_dataset, process_id):
    """
    Learns a line from the given ordinal constraints
    """
    best_embedding = None
    best_violated_constraints = float("inf")
    num_buckets = int(1 / EPSILON)

    for i in tqdm(range(num_points), position=process_id * 2 + BAR_POSITION_OFFSET, leave=False,
                  desc=f"[Core {process_id}] Points     "):
        # find constraints where p_i is the first
        base_point = i + process_id * num_points
        constraints = list(filter(lambda x: x[0] == base_point, idx_constraints))
        G = build_graph_from_constraints(constraints, base_point)
        reoriented = feedback_arc_set(G, process_id=process_id)
        try:
            topological_ordered_nodes = list(nx.topological_sort(reoriented))
        except nx.exception.NetworkXUnfeasible:
            # Skip iteration if a topological ordering cannot be built
            continue

        buckets = create_buckets(topological_ordered_nodes, num_buckets)

        if not buckets[0]:
            # Skip iteration if the point had no constraints to begin with
            continue

        # From here I have to build the WLCC instance
        representatives, base_embedding = build_representative_embedding(base_point, buckets)
        base_embedding = patch_embedding_from_all_constraints(base_embedding, representatives,
                                                              all_dataset)

        base_embedding, projected_constraints, constraints_weights = create_wlcc(base_embedding, all_dataset)
        base_weight_violated_constraints = count_violated_constraints(base_embedding, projected_constraints,
                                                                      representatives, constraints_weights)

        embedding, new_representatives, violated_constraints = search_better_embedding(projected_constraints,
                                                                                       best_embedding,
                                                                                       best_violated_constraints,
                                                                                       base_point,
                                                                                       representatives, base_embedding,
                                                                                       base_weight_violated_constraints,
                                                                                       constraints_weights,
                                                                                       process_id=process_id)

        if violated_constraints < best_violated_constraints:
            best_embedding = embedding.copy()
            best_violated_constraints = violated_constraints

    return best_embedding, best_violated_constraints
