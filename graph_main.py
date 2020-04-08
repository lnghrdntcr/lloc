## Idea, since we build a graph at some point
#  how does the algorithm behave when we feed a graph?
from utils import load_graph, graph_format_arguments, save_embedding
from llcc import graph_llcc
from multiprocessing import Pool
import multiprocessing

best_embedding = {}
best_violated_constraints = float("inf")

if __name__ == "__main__":
    graph = load_graph("./datasets/graphs/facebook_combined.txt")

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
    print("Done!")