from utils import load_graph, graph_format_arguments, save_embedding, maps_to, reverse_edges
from llcc import graph_llcc
from multiprocessing import Pool
import multiprocessing
from config import GRAPH_MOCK, EPSILON, GRAPH_NUM_NODES
import networkx as nx
from IPython import embed

best_embedding = {}
best_violated_constraints = float("inf")

if __name__ == "__main__":
    if GRAPH_MOCK:
        graph = nx.to_directed(nx.newman_watts_strogatz_graph(GRAPH_NUM_NODES, 7, 0.1))
    else:
        graph = load_graph("./datasets/graphs/facebook_combined.txt")

    pagerank = [key for key, _ in sorted(nx.pagerank(graph).items(), key=lambda x: x[1])]
    top_x = list(map(int, pagerank[0:10]))

    cpu_count = multiprocessing.cpu_count()
    process_pool = Pool(cpu_count)
    arguments = graph_format_arguments(graph, cpu_count)

    responses = process_pool.starmap(graph_llcc, arguments)

    for embedding, n_violated_constraints in responses:
        if n_violated_constraints < best_violated_constraints:
            best_embedding = embedding
            best_violated_constraints = n_violated_constraints

    print(
        f"Best embedding with {best_violated_constraints} errors over {len(graph.edges)} constraints. Max possible constraints -> {len(graph.nodes) ** 2 / 2} ")

    save_embedding(embedding)

    process_pool.close()

    print("DONE! Checking compatibility with pagerank...")
    for embeds_to in range(int(1 // EPSILON + 1)):
        count = 0
        for el in maps_to(embedding, embeds_to):
            if el in top_x:
                count += 1
        print(f"Compatibility with bucket {embeds_to} -> {count / len(top_x)}")
    print("DONE! Exiting...")
    exit(0)
