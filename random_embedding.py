from format_dataset import format_mnist_from_labels
from utils import train_test_split
import networkx as nx
from itertools import combinations
from random import random as rand
import numpy as np

if __name__ == "__main__":
    print("RANDOM EMBEDDING")

    idx_constraints, reverse_cache, (x, y), class_distr = format_mnist_from_labels()
    G = nx.DiGraph()

    train, test = train_test_split(idx_constraints, test_percentage=0.3)

    for constraint in train:
        for u, v in list(combinations(constraint, 2))[:-1]:
            G.add_edge(u, v)

    random_embedding = {}
    missing = 0
    for node in G.nodes:
        random_embedding[node] = rand()
    error_rate = 0
    for test_constraint in test:
        i, j, k = test_constraint
        new_cost = 0
        if ((f_i := random_embedding.get(i)) is not None) and ((f_j := random_embedding.get(j)) is not None) and (
                (f_k := random_embedding.get(k)) is not None):
            new_cost += int(np.abs(f_i - f_j) > np.abs(f_i - f_k))
        else:
            new_cost += 1
            missing += 1

        if new_cost != 0:
            error_rate += 1

    print(f"Random embedding: Accuracy = {100 - round((error_rate / len(test)) * 100, 4)}%\n")

