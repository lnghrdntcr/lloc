import random
from itertools import combinations

import networkx as nx
import numpy as np
from tqdm import tqdm

from config import EPSILON


def feedback_arc_set(G):
    """
    Reorient edges incrementally to build a DAG
    :param G: Input Directed graph
    :return: A directed acyclic graph built from G
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
                cur_cycles = list(nx.simple_cycles(ret))

                # Early exit condition, there are no more cycles
                # so its pointless to continue to remove edges
                if not cur_cycles:
                    return ret.copy()
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


def format_embedding(base, embedding, mapped_to_representatives, move_pattern=None):
    ret = {
        str(base): 0
    }

    for i, el in enumerate(embedding):
        ret[str(el)] = i + 1

    for key, value in mapped_to_representatives.items():

        if move_pattern is not None:
            if value == move_pattern[0]:
                ret[key] = move_pattern[1]
            elif value == move_pattern[1]:
                ret[key] = move_pattern[0]
        else:
            ret[key] = value

    return ret


def count_violated_constraints(embedding, constraints, buckets=False):
    count = 0
    for constraint in constraints:
        unrolled_constraints = combinations(constraint, 2)
        for unrolled_constraint in unrolled_constraints:
            assert len(unrolled_constraint) == 2
            s, t = unrolled_constraint
            f_s = embedding.get(str(s))
            f_t = embedding.get(str(t))
            if (f_s is not None) and (f_t is not None):
                count += int(f_s > f_t)
            elif not buckets:
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
        for idx in constraint:
            if not representative_map.get(str(idx)) and idx not in representatives:
                # Map to bucket 0 every element missing from the map
                ret[str(idx)] = 0

    return ret


def graph_patch_representative_map_from_all_constraints(representative_map, graph):
    ret = representative_map.copy()
    for node in graph.nodes:
        if not representative_map.get(str(node)):
            ret[str(node)] = 0
    return ret


def build_representative_embedding(buckets):
    representatives = []
    representatives_map = {}

    for idx, bucket in enumerate(buckets):
        representatives.append(random.choice(bucket))

        for el in bucket:
            if el != representatives[idx]:
                representatives_map[str(el)] = idx + 1
    return representatives, representatives_map


def search_better_embedding(all_dataset, current_best_embedding, current_best_violated_constraints, point_id,
                            process_id, representatives, representatives_map):
    for i in tqdm(range(int(1 / EPSILON) - 1), position=process_id * 2 + 1, leave=False,
                  desc=f"[Core {process_id}] Embeddings"):
        next_representatives = representatives.copy()
        next_representatives[i], next_representatives[i + 1] = next_representatives[i + 1], next_representatives[i]
        cur_embedding = format_embedding(point_id, representatives, representatives_map)
        next_embedding = format_embedding(point_id, next_representatives, representatives_map, move_pattern=(i, i + 1))

        cur_violated_constraints = count_violated_constraints(cur_embedding, all_dataset)
        next_violated_constraints = count_violated_constraints(next_embedding, all_dataset)

        if cur_violated_constraints > next_violated_constraints:
            representatives = next_representatives
            representatives_map = next_embedding

    embedding = format_embedding(point_id, representatives, representatives_map)
    violated_constraints = count_violated_constraints(embedding, all_dataset)

    if violated_constraints < current_best_violated_constraints:
        current_best_embedding = embedding
        current_best_violated_constraints = violated_constraints

    return current_best_embedding, current_best_violated_constraints


def llcc(idx_constraints, num_points, all_dataset, process_id):
    """
    Learns a line from the given constraints
    """
    best_embedding = {}
    best_violated_constraints = float("inf")
    for i in tqdm(range(num_points), position=process_id * 2, leave=False, desc=f"[Core {process_id}] Points    "):
        # find constraints where p_i is the first
        point_id = i + process_id * num_points
        constraints = list(filter(lambda x: x[0] == point_id, idx_constraints))
        G = build_graph_from_constraints(constraints, point_id)
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

        representatives_map = patch_representative_map_from_all_constraints(representatives_map, representatives,
                                                                            all_dataset)

        embedding, violated_constraints = search_better_embedding(all_dataset, best_embedding,
                                                                  best_violated_constraints, point_id, process_id,
                                                                  representatives, representatives_map)

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


def pagerank_llcc(idx_constraints, num_points, all_dataset, process_id):
    best_embedding = {}
    best_violated_constraints = float("inf")

    G = build_graph_from_triplet_constraints(all_dataset)
    reoriented = feedback_arc_set(G)
    for i in tqdm(range(num_points), position=process_id * 2, leave=False, desc=f"[Core {process_id}] Points    "):
        point_id = i + process_id * num_points

        pr = nx.pagerank(reoriented, personalization={point_id: 1})

        sorted_pr = [str(k) for k, _ in sorted(pr.items(), key=lambda x: x[1])]
        embedding = dict([(v, i) for i, v in enumerate(sorted_pr)])
        violated_constraints = count_violated_constraints(embedding, all_dataset)

        if violated_constraints < best_violated_constraints:
            best_embedding = embedding
            best_violated_constraints = violated_constraints

    return best_embedding, best_violated_constraints


def graph_llcc(nodes, graph: nx.DiGraph, process_id):
    best_embedding = {}
    best_violated_constraints = float("inf")

    for node in tqdm(nodes, position=process_id * 2, leave=False, desc=f"[Core {process_id}] Points    "):
        temp_graph = graph.copy()
        temp_graph.remove_node(node)

        reoriented = feedback_arc_set(temp_graph)

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

        representatives_map = graph_patch_representative_map_from_all_constraints(representatives_map, graph)

        # Perform exhaustive enumeration among all representatives
        embedding, violated_constraints = graph_bfs_search_best_embedding(best_embedding,
                                                                          best_violated_constraints, graph,
                                                                          node, process_id, representatives,
                                                                          representatives_map)

        if violated_constraints < best_violated_constraints:
            best_embedding = embedding
            best_violated_constraints = violated_constraints

    return best_embedding, best_violated_constraints


def graph_search_better_embedding(current_best_embedding, current_best_violated_constraints, graph, node, process_id,
                                  representatives,
                                  representatives_map):
    for i in tqdm(range(int(1 / EPSILON) - 1), position=process_id * 2 + 1, leave=False,
                  desc=f"[Core {process_id}] Embeddings"):

        next_representatives = representatives.copy()
        next_representatives[i], next_representatives[i + 1] = next_representatives[i + 1], next_representatives[i]
        cur_embedding = format_embedding(node, representatives, representatives_map)
        next_embedding = format_embedding(node, next_representatives, representatives_map, move_pattern=(i, i + 1))

        cur_violated_constraints = graph_count_violated_constraints(cur_embedding, graph)
        next_violated_constraints = graph_count_violated_constraints(next_embedding, graph)

        if cur_violated_constraints > next_violated_constraints:
            representatives = next_representatives
            representatives_map = next_embedding

    embedding = format_embedding(node, representatives, representatives_map)
    violated_constraints = graph_count_violated_constraints(embedding, graph)

    if violated_constraints < current_best_violated_constraints:
        current_best_embedding = embedding
        current_best_violated_constraints = violated_constraints

    return current_best_embedding, current_best_violated_constraints


def graph_bfs_search_best_embedding(best_embedding, best_violated_constraints, graph, node, process_id, representatives,
                                    representatives_map):
    best_representatives = representatives.copy()
    best_representatives_map = representatives_map.copy()
    swapped_positions = []
    cursor = 0

    for i in tqdm(range(int(1 / EPSILON)), position=process_id * 2 + 1, leave=False,
                  desc=f"[Core {process_id}] Embeddings"):
        # every point must become the "first"
        violated_constraints = []
        for j in range(cursor, len(representatives)):
            r = best_representatives.copy()
            rm = best_representatives_map.copy()
            r[cursor], r[j] = r[j], r[cursor]

            temp_embedding = format_embedding(node, r, rm, move_pattern=(j, cursor))
            violated_constraints.append(graph_count_violated_constraints(temp_embedding, graph))

        # To have a decrease in violated constraints
        # I have to swap element in position move_to to element to position cursor
        move_to = np.argmin(violated_constraints)
        if move_to != 0:
            best_representatives[move_to + cursor], best_representatives[cursor] = best_representatives[cursor], \
                                                                                   best_representatives[
                                                                                       move_to + cursor]
            swapped_positions.append((cursor + 1, cursor + move_to + 1))
            for k in best_representatives_map.keys():
                if best_representatives_map[k] == move_to + cursor + 1:
                    best_representatives_map[k] = cursor + 1
                elif best_representatives_map[k] == cursor + 1:
                    best_representatives_map[k] = move_to + cursor + 1

        cursor += 1

    final_embedding = format_embedding(node, best_representatives, best_representatives_map)
    n_violated_constraints = graph_count_violated_constraints(final_embedding, graph)

    if n_violated_constraints < best_violated_constraints:
        best_embedding = final_embedding
        best_violated_constraints = n_violated_constraints

    return best_embedding, best_violated_constraints
