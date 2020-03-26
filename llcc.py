import random
from itertools import combinations
import networkx as nx


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


def format_embedding(base, embedding, mapped_to_representatives):
    # TODO: I should map every image not in the embedding to class 0
    ret = {
        str(base): 0
    }

    for i, el in enumerate(embedding):
        ret[str(el)] = i + 1

    for key, value in mapped_to_representatives.items():
        ret[key] = value

    return ret


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


def patch_representative_map_from_all_constraints(representative_map, all_idxs):
    ret = representative_map.copy()
    for constraint in all_idxs:
        assert len(constraint) == 3
        for idx in constraint:
            if not representative_map.get(str(idx)):
                # Map to bucket 0 every element missing from the map
                ret[str(idx)] = 0

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
