import json
from io import BytesIO
from math import factorial
from math import floor
from os import system, mkdir
from time import time

import networkx as nx
import numpy as np
import requests as req
from PIL import Image
from tqdm import tqdm

from config import EPSILON, MNIST_COL_SIZE, MNIST_ROW_SIZE
from llcc import feedback_arc_set, build_graph_from_triplet_constraints


def format_arguments(points, num_points, cpu_count, class_distribution, use_pagerank=False):
    print("Formatting arguments...", end="")
    chunk_size = floor(num_points / cpu_count)
    arguments = []

    if use_pagerank:
        G = build_graph_from_triplet_constraints(points)
        print("\nPrecomputing Graph...")
        reoriented = feedback_arc_set(G)

    for i in range(cpu_count - 1):
        projected_datapoints = filter(lambda x: i * chunk_size <= x[0] < (i + 1) * chunk_size, points)
        dp = list(projected_datapoints)
        if use_pagerank:
            arguments.append((dp, chunk_size, points.copy(), i, reoriented))
        else:
            arguments.append((dp, chunk_size, points.copy(), i, class_distribution))

    # Last points are handled separately
    dp = list(filter(lambda x: (cpu_count - 1) * chunk_size <= x[0], points))

    if use_pagerank:
        arguments.append((dp, num_points - (chunk_size * (cpu_count - 1)), points, cpu_count - 1, reoriented))
    else:
        arguments.append((dp, num_points - (chunk_size * (cpu_count - 1)), points, cpu_count - 1, class_distribution))

    print("done!")
    return arguments


def graph_format_arguments(graph, cpu_count):
    print("Formatting arguments...", end="")
    chunk_size = floor(len(graph.nodes) / cpu_count)
    arguments = []
    nodes = list(graph.nodes)
    for i in range(cpu_count - 1):
        arguments.append((nodes[i * chunk_size: (i + 1) * chunk_size].copy(), graph.copy(), i))

    arguments.append((nodes[(cpu_count - 1) * chunk_size:].copy(), graph.copy(), cpu_count - 1))
    print("done!")
    return arguments


def reverse_edges(G):
    ret = nx.DiGraph()

    for u, v in G.edges:
        ret.add_edge(v, u)

    return ret


def save_mnist_image(image_array, label, idx, correlation=False, bucketing=True, image_name=0):
    cur_image = Image.fromarray(image_array.reshape((MNIST_ROW_SIZE, MNIST_COL_SIZE)).astype(np.uint8))

    if bucketing:
        cur_image.save(f"./results/mnist/{label}/{image_name}.png")
    else:
        cur_image.save(f"./results/mnist/new_algo/{image_name}.png")


def save_fec_results(embeddings, image_cache, crop_map, directory=True):
    idx = 0
    for image_index, directory in tqdm(embeddings.items()):
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
            # if directory:
            #    face.save(f"./results/FEC/{directory}/{image_index}.jpg")
            # else:
            face.save(f"./results/FEC/pagerank/{idx}.jpg")
            idx += 1


def n_choose_k(n, k):
    assert n > k
    num = factorial(n)
    den = factorial(k) * factorial(n - k)

    return num / den


def setup_results_directories(dataset):
    system(f"cp -r results/{dataset} results/backup/{dataset}-{time()}")
    system(f"rm -rf results/{dataset}/*")
    # For bucketing version
    for i in range(int(1 / EPSILON) + 1):
        mkdir(f"./results/{dataset}/{i}", )

    # For pagerank version
    mkdir(f"./results/{dataset}/pagerank")
    mkdir(f"./results/{dataset}/new_algo")


def pretty_print_embedding(embedding):
    pretty_print = ", ".join([f"{el} -> {idx + 1}" for idx, el in enumerate(embedding)])
    print(f"Current embedding into R^1 = {pretty_print}")


def load_graph(path):
    return nx.read_edgelist(path, create_using=nx.DiGraph)


def save_embedding(embedding):
    print("Saving...", end="")
    for k, v in embedding.items():
        embedding[k] = int(v)
    embedding_string = json.dumps(embedding)
    with open(f"./results/graphs/facebook-{str(int(1 // EPSILON))}.json", "w+") as savefile:
        savefile.write(embedding_string)

    print("done!")


def maps_to(embedding, index):
    vertices = []
    for k, v in embedding.items():
        if v == index:
            vertices.append(int(k))
    return vertices


def map_class_distribution_to_bucket_distribution(class_distribution):
    num_buckets = int(1 / EPSILON)
    bucketed_class_distribution = [0 for _ in range(num_buckets)]
    elements_to_aggregate = len(class_distribution) // num_buckets
    i = 0
    for _ in range(num_buckets - 1):
        acc = 0
        for j in range(i, i + elements_to_aggregate):
            acc += class_distribution[j]

        bucketed_class_distribution[i // elements_to_aggregate] = acc
        i += elements_to_aggregate

    bucketed_class_distribution[-1] = 1 - sum(bucketed_class_distribution)
    return bucketed_class_distribution


def select_bucket_from_embedding_value(value, class_distr):
    most_frequent_class = np.max(class_distr)

    for i in range(int(1 // EPSILON)):
        scaling_factor = most_frequent_class / class_distr[i]
        if i - 0.1 * scaling_factor <= value <= i + 0.1 * scaling_factor:
            return i
    return int(1 // EPSILON)
