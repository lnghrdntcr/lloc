from itertools import combinations

import numpy as np
import networkx as nx
from IPython import embed
from scipy.sparse import csgraph
from scipy.spatial.distance import cosine


from config import EPSILON, MNIST_ERROR_RATE, MNIST_DIGIT_EXCLUSION_PROBABILITY
from llcc import count_violated_constraints

from format_dataset import format_mnist_from_labels

# Create Matrix representing the constraints, how? Create a DiGraph where for each constraint [a, b, c]
# there exists an edge (a, b), (a, c), (b, c). For each pair of nodes (i, j)
# assess whether there exists a directed path from i to j.
# If so A_{i,j} = 1
# WRONG IDEA, in the case of a `dense` problem, the matrix is complete...
# New Idea, Construct the graph as before but use the adjacency matrix!
# New Idea, the graph is useless and build the affinity matrix from the dataset
# Final Idea, Rollback to the initial idea to create graph from pairwise constraints.
# Used csgraph.laplacian to get Laplacian matrix

if __name__ == "__main__":
    print(f"Running test with EPSILON={EPSILON}, MNIST_ERROR_RATE={MNIST_ERROR_RATE}, MNIST_DIGIT_EXCLUSION_PROBABILITY={MNIST_DIGIT_EXCLUSION_PROBABILITY}")

    idx_constraints, reverse_cache, (x, y), class_distr = format_mnist_from_labels(use_distance=False)
    G = nx.DiGraph()
    similarity_matrix = np.zeros((len(x), len(x)))
    new_constraints = []
    for i in range(len(x)):
        for j in range(len(x)):
            similarity_matrix[i, j] = 1 - cosine(x[i], x[j])

    for i in range(len(x)):
        for j in range(len(x)):
            for k in range(len(x)):
                if i == j or i == k or j == k:
                    continue

                if similarity_matrix[i, j] > similarity_matrix[i, k]:
                    new_constraints.append([i, j, k])

    for constraint in new_constraints:
        edges = combinations(constraint, 2)
        for u, v in edges:
            G.add_edge(u, v)

    # Create adjacency matrix in dense format in order to retrieve L in dense format
    M = nx.adjacency_matrix(G).todense()
    L = csgraph.laplacian(M, normed=False)

    # Calculate eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(L)

    for i in range(len(eigenvalues)):
        # Get the embedding (eigenvector) associated to the second highest eigenvalue
        index = [(idx, value) for idx, value in enumerate(sorted(eigenvalues))][-i][0]
        indexed_eigenvector = [(k, v) for k, v in enumerate(eigenvectors[index])]
        sorted_embedding = sorted(indexed_eigenvector, key=lambda x: x[1])
        embedding = dict([(v[0], k) for k, v in enumerate(sorted_embedding)])

        # Count the violated constraints
        violated_constraints = count_violated_constraints(embedding, idx_constraints)
        print(f"Violated constraints using the {i}th highest eigenvector -> {violated_constraints / len(idx_constraints) * 3}%")

    embed()
