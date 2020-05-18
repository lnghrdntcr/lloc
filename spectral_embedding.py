from itertools import combinations

import numpy as np
import networkx as nx
from IPython import embed
from scipy.sparse import csgraph
from scipy.spatial.distance import cosine
from random import random as rand

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
from utils import train_test_split

if __name__ == "__main__":

    idx_constraints, reverse_cache, (x, y), class_distr = format_mnist_from_labels()
    G = nx.DiGraph()

    train, test = train_test_split(idx_constraints, test_percentage=0.5)

    for constraint in train:
        for u,v in list(combinations(constraint, 2))[:-1]:
            G.add_edge(u, v)

    # Create adjacency matrix in dense format in order to retrieve L in dense format
    M = nx.adjacency_matrix(G).todense()
    L = csgraph.laplacian(M, normed=False)

    # Calculate eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(L)

    for ii in range(len(eigenvalues)):
        # Get the embedding (eigenvector) associated to the second highest eigenvalue
        index = [(idx, value) for idx, value in enumerate(sorted(eigenvalues))][-ii][0]
        indexed_eigenvector = [(k, v) for k, v in enumerate(eigenvectors[index])]
        sorted_embedding = sorted(indexed_eigenvector, key=lambda x: x[1])
        best_embedding = dict([(v[0], k) for k, v in enumerate(sorted_embedding)])

        # Count the violated constraints
        #violated_constraints = count_violated_constraints(best_embedding, idx_constraints, [1 for _ in range(len(idx_constraints))], use_distance=True)
        #print(f"Violated constraints using the {i}th highest eigenvector -> {violated_constraints / len(idx_constraints) * 3}%")

        error_rate = 0
        missing = 0
        for test_constraint in test:
            i, j, k = test_constraint
            new_cost = 0
            f_i = best_embedding.get(i)
            f_j = best_embedding.get(j)
            f_k = best_embedding.get(k)
            if (f_i is not None) and (f_j is not None) and (f_k is not None):
                new_cost += int(np.abs(f_i - f_j) > np.abs(f_i - f_k))
            else:
                new_cost += 1
                missing  += 1

            if new_cost != 0:
                error_rate += 1

        print(f"{ii}th highest eigenvalue: Precision = {100 - round((error_rate / len(test)) * 100, 4)}\n")

