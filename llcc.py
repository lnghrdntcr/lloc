import random
from collections import OrderedDict
from itertools import combinations

import networkx as nx
import numpy as np
from tqdm import tqdm

from config import EPSILON


def feedback_arc_set(G: nx.DiGraph, process_id=0):
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


def old_feedback_arc_set(G, bar=False, process_id=0):
    """
    Reorient edges incrementally to build a DAG
    :param G: Input Directed graph
    :return: A directed acyclic graph built from G
    """
    ret = G.copy()

    for node in ret.nodes:
        try:
            ret.remove_edge(node, node)
        except:
            continue

    cycles = nx.simple_cycles(ret)
    edges_removed = 0

    if bar:
        cycles_iterator = tqdm(cycles, position=process_id * 2 + 1, leave=False, desc=f"[Core {process_id}] FAS")
    else:
        cycles_iterator = cycles

    for cycle in cycles_iterator:
        cycle_generating_vertices = list(combinations(cycle, 2))
        for vertices in cycle_generating_vertices:
            u, v = vertices
            try:
                ret.remove_edge(u, v)
                ret.add_edge(v, u)
                edges_removed += 1
                cur_cycles = nx.simple_cycles(ret)

                # Early exit condition, there are no more cycles
                # so its pointless to continue to remove edges
                try:
                    next(cur_cycles)
                except StopIteration:
                    return ret.copy()
            except nx.exception.NetworkXError:
                pass
    return ret.copy()


def get_buckets(arr, num_buckets, bucketed_class_distribution):
    """
    Does bucketing respecting class distribution
    :param arr:
    :param num_buckets:
    :param bucketed_class_distribution:
    :return:
    """
    # IDEA:
    # I want the size of the interval from which I'm embedding the points to be inversly proportional to the class distribution
    # for instance -> if all elements are distributed uniformly the buckets should be somewhat contiguos and the range is (0.5 - base, 0.5 + base)
    # but if the presence of a class is half of the average I want to make overlaps possible -> for instance if the class of 1 is 0.5 of the size of the maximum class
    # I want the interval to be twice as big!
    # So it should be something like (0.5 * scale_factor - base, 0.5 * scale_factor + base)

    buckets = []
    bucketing_factor = int(len(arr) // num_buckets)

    base_idx = 0
    for elements_distribution in bucketed_class_distribution[:-1]:
        until_idx = base_idx + int(elements_distribution * len(arr))
        buckets.append(arr[base_idx: until_idx])
        base_idx = until_idx

    buckets.append(arr[base_idx:])
    return buckets


def format_embedding(base, embedding, mapped_to_representatives, class_distribution, move_pattern=None):
    ret = OrderedDict()
    ret[str(base)] = 0

    assert len(class_distribution) == len(embedding)

    most_frequent_class = np.max(class_distribution)

    for i, el in enumerate(embedding):
        ret[str(el)] = i + 1

    # skip elements that are mapped to zero
    for key, maps_to in list(ret.items())[1:]:
        # get all_elements that are mapped to `maps_to`
        bucket_elements = list(filter(lambda x: x[1] == maps_to, mapped_to_representatives.items()))
        # Note: class distribution changes as the elements of the embedding are swapped
        scale_factor = most_frequent_class / class_distribution[int(maps_to) - 1]
        # now i map all of those elements from (0.5 - maps_to, 0.5 + maps_to) uniformly at random
        # note: if move_pattern != None the only thing I need to change is the element_maps_to value
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


def count_violated_constraints(embedding, constraints, constraints_weights, ignore_missing_values=False):
    count = 0
    for idx, constraint in enumerate(constraints):
        s, t, k = constraint
        f_i = embedding.get(str(s))
        f_j = embedding.get(str(t))
        f_k = embedding.get(str(k))
        if (f_i is not None) and (f_j is not None) and (f_k is not None):
            count += int(np.abs(f_i - f_j) >= np.abs(f_i - f_k)) * constraints_weights[idx]
        elif not ignore_missing_values:
            count += 1

    return count


def graph_count_violated_constraints(embedding, graph: nx.DiGraph, buckets=False):
    errors = 0
    for u, v in graph.edges:
        f_u = embedding.get(str(u))
        f_v = embedding.get(str(v))

        if (f_u is not None) and (f_v is not None):
            errors += int(f_u > f_v)
        elif not buckets:
            errors += 1

    return errors


def build_graph_from_constraints(constraints, cur_vertex):
    """
    Builds the tournament from the given constraints
    :param constraints: Triplet Constraints
    :return: The built turnament
    """
    G = nx.DiGraph()
    edges = list(map(lambda x: (x[1], x[2]), constraints))
    nodes = set([item for sublist in constraints for item in sublist if item != cur_vertex])
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def patch_representative_map_from_all_constraints(representative_map, representatives, all_idxs):
    ret = representative_map.copy()
    for constraint in all_idxs:
        assert len(constraint) == 3
        for point in constraint:
            if not representative_map.get(str(point)):
                if point not in representatives:
                    # Map to bucket 0 every element missing from the map
                    ret[str(point)] = 0
                else:
                    ret[str(point)] = representatives.index(point)

    return ret

def build_representative_embedding(buckets):
    representatives = []
    representatives_map = OrderedDict()

    for idx, bucket in enumerate(buckets):
        representatives.append(random.choice(bucket))

        for el in bucket:
            if el != representatives[idx]:
                representatives_map[str(el)] = idx + 1
    return representatives, representatives_map


def search_better_embedding(all_dataset, current_best_embedding, current_best_violated_constraints, point_id,
                            process_id, representatives, representatives_map, class_distribution, constraints_weight):
    local_best_embedding = None
    local_best_violated_constraints = float("inf")

    for i in tqdm(range(int(1 / EPSILON) - 1), position=process_id * 2 + 1, leave=False,
                  desc=f"[Core {process_id}] Embeddings "):
        next_representatives = representatives.copy()
        next_representatives[i], next_representatives[i + 1] = next_representatives[i + 1], next_representatives[i]
        next_class_distribution = class_distribution.copy()
        next_class_distribution[i], next_class_distribution[i + 1] = next_class_distribution[i + 1], \
                                                                     next_class_distribution[i]

        if local_best_embedding is None:
            local_best_embedding = format_embedding(point_id, representatives, representatives_map, class_distribution)

        next_embedding = format_embedding(point_id, next_representatives, representatives_map, next_class_distribution,
                                          move_pattern=(i, i + 1))

        if local_best_violated_constraints == float("inf"):
            local_best_violated_constraints = count_violated_constraints(local_best_embedding, all_dataset,
                                                                         constraints_weight)

        next_violated_constraints = count_violated_constraints(next_embedding, all_dataset, constraints_weight)

        if local_best_violated_constraints > next_violated_constraints:
            representatives = next_representatives
            representatives_map = next_embedding.copy()
            local_best_embedding = next_embedding.copy()
            local_best_violated_constraints = next_violated_constraints
            class_distribution = next_class_distribution

    embedding = local_best_embedding.copy()
    violated_constraints = local_best_violated_constraints

    if violated_constraints < current_best_violated_constraints:
        current_best_embedding = embedding
        current_best_violated_constraints = violated_constraints

    return current_best_embedding, current_best_violated_constraints


def create_wlcc(representatives_map, all_dataset):
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
        i, j, j = constraint
        if np.abs(representatives_map[str(i)] - representatives_map[str(j)]) >= np.abs(
                representatives_map[str(i)] - representatives_map[str(k)]):
            weights.append(count)
        else:
            weights.append(0)

    new_idx_constraints = [constraint for constraint, count in new_constraints.items()]
    assert len(new_idx_constraints) == len(weights)
    return representatives_map, new_idx_constraints, weights


def llcc(idx_constraints, num_points, all_dataset, process_id):
    """
    Learns a line from the given constraints
    """
    best_embedding = OrderedDict()
    best_violated_constraints = float("inf")
    num_buckets = int(1 / EPSILON)
    class_distribution = [1 / num_buckets for _ in range(num_buckets)]
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

        buckets = get_buckets(topological_ordered_nodes, num_buckets, class_distribution)

        if not buckets[0]:
            # Skip iteration if the point had no constraints to begin with
            continue

        # From here I have to build the WLCC instance
        representatives, representatives_map = build_representative_embedding(buckets)
        representatives_map = patch_representative_map_from_all_constraints(representatives_map, representatives,
                                                                            all_dataset)

        representatives_map, projected_constraints, constraints_weights = create_wlcc(representatives_map, all_dataset)

        embedding, violated_constraints = search_better_embedding(projected_constraints, best_embedding,
                                                                  best_violated_constraints, point_id, process_id,
                                                                  representatives, representatives_map,
                                                                  class_distribution, constraints_weights)

        if violated_constraints < best_violated_constraints:
            best_embedding = embedding.copy()
            best_violated_constraints = violated_constraints

    return best_embedding, best_violated_constraints


def build_graph_from_triplet_constraints(idx_constraints):
    G = nx.DiGraph()

    for constraint in idx_constraints:
        edges = combinations(constraint, 2)
        for edge in edges:
            G.add_edge(edge[0], edge[1])

    return G.copy()


def pagerank_llcc(_, num_points, all_dataset, process_id, G):
    best_embedding = OrderedDict()
    best_violated_constraints = float("inf")

    for i in tqdm(range(num_points), position=process_id * 2, leave=False, desc=f"[Core {process_id}] Points    "):
        point_id = i + process_id * num_points

        pr = nx.pagerank(G, personalization={point_id: 1})

        sorted_pr = [str(k) for k, _ in sorted(pr.items(), key=lambda x: x[1])]
        embedding = dict([(v, i) for i, v in enumerate(sorted_pr)])
        violated_constraints = count_violated_constraints(embedding, all_dataset)

        if violated_constraints < best_violated_constraints:
            best_embedding = embedding
            best_violated_constraints = violated_constraints

    return best_embedding, best_violated_constraints

