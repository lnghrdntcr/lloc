import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
from lloc import count_raw_violated_constraints
import multiprocessing
from multiprocessing import Pool
from time import sleep, time
from scipy.special import comb
from random import random as thread_safe_random
import sys

N_POINTS = 100
sns.set_style("whitegrid")

RESOLUTION = 1 / 10

def clear_figures(): 
    plt.clf()
    plt.cla()
    plt.close()


def random_embedding(n_points, dim=2):
    return np.random.rand(n_points, dim)


def plotline(line):
    coeff = line[1] / line[0]
    axis = np.linspace(0, 1, N_POINTS)
    y = [coeff * x for x in axis]

    plt.plot(axis, y)

    # return [(x, coeff*x) for x in np.linspace(0, 1, N_POINTS)]


def get_x_y(arr):
    return [x[0] for x in arr], [y[1] for y in arr]


def avg(arr):
    return sum(arr) / len(arr)


def build_constraints_set(points):
    n_points = len(points)
    constraints = []
    distance_matrix = np.zeros((n_points, n_points))

    for i, j in tqdm(combinations(range(n_points), 2), desc="Distance generation..."):
        distance = np.linalg.norm(points[i] - points[j])
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

    for i in tqdm(range(n_points), desc="Triplet generation..."):
        for j in range(n_points):

            # Skip trivial constraints
            if i == j:
                continue

            for k in range(n_points):
                if distance_matrix[i, j] < distance_matrix[i, k]:
                    constraints.append([i, j, k])

    return distance_matrix, constraints


def project(points, project_to):
    ret = []

    def _project(vect):
        return np.matmul(np.matmul(vect, project_to.T) / np.matmul(project_to, project_to.T), project_to)

    for point in points:
        ret.append(_project(point))

    return np.array(ret)


def get_random_line(angle = None):
    if angle:
        v = np.array([np.cos(angle), np.sin(angle)])
    else:
        x, y = np.random.normal(), np.random.normal()
        v = np.array([x, y])
        v /= np.linalg.norm(v)

    return v.reshape((1, 2))


def test_line_embedding(base_embedding, constraints, core_id, seed, range_begin, range_end):
    violated_constraints_trace = []
    np.random.seed(seed)

    for angle_base in range(int(range_begin / RESOLUTION), int(range_end / RESOLUTION)):
        angle = (angle_base * RESOLUTION - 180) / 180 * np.pi # radians
        random_line = get_random_line(angle=angle)
        projected = project(base_embedding, random_line).reshape(N_POINTS, 2)
        violated_constraints = count_raw_violated_constraints(projected, constraints)
        violated_constraints_trace.append(violated_constraints)
        sys.stdout.write(
            f"\r[CORE {core_id}] Violated {violated_constraints} constraints. Percentage = {violated_constraints / len(constraints) * 100}%. angle = {angle / np.pi * 180}")
        sys.stdout.flush()
        with open("./results/results_2d_to_1d_angle.csv", "a+") as f:
            f.write(f"{violated_constraints},{len(constraints)},{angle}\n")

    return violated_constraints_trace


def main():
    print(f"Running test with {N_POINTS} points.")
    base_embedding = random_embedding(N_POINTS)
    # Prepare for parallel test execution
    cpu_count = multiprocessing.cpu_count()
    process_pool = Pool(cpu_count)
    distance_matrix, constraints = build_constraints_set(base_embedding)
    all_violated_constraints = []
    random_seeds = []
    print(f"Generating random seeds...")
    for i in range(cpu_count):
        # Sleep for a random time in order to prevent
        # the rng to choose the same random line across cores!
        sleep(thread_safe_random())
        random_seeds.append(int(thread_safe_random() * 10000000))

    ranges = [(i * 360 / cpu_count, (i + 1) * 360 / cpu_count) for i in range(cpu_count)]
    print(f"Testing....")
    begin = time()
    results = process_pool.starmap(test_line_embedding, [(base_embedding, constraints, i, random_seeds[i], ranges[i][0], ranges[i][1]) for i in range(cpu_count)])
    print(f"Testing finished. Time elapsed = {time() - begin}s")
    for result in results:
        all_violated_constraints.extend(result)
    average_violated_constraints = avg(all_violated_constraints)
    print(
        f"Violating {average_violated_constraints} constraints on average. Percentage = {average_violated_constraints / len(constraints) * 100}%")


if __name__ == "__main__":
    points_map = dict([(int(i * comb(i - 1, 2)), i) for i in [j * 25 for j in range(1,8)]])
    for N_POINTS in [25 + i for i in range(1, 75)]:
        main()

    # df = pd.read_csv("./results/results_2d_to_1d.csv")
    #
    # df["perc"] = df["violated_constraints"] / df["total_constraints"]
    #
    # df["points"] = df["total_constraints"].apply(lambda x: points_map[x])
    #
    # print(df["perc"].std() / df["perc"].mean() * 100, "%")

    # maxes = {
    #     "x": [],
    #     "y": []
    # }
    # for i, group in df.groupby('total_constraints'):
    #     sns.distplot(group["perc"], kde=True, hist=False, label=f"{points_map[i]}")
    #     plt.legend()
    #     plt.xlim((0, 1 / 3 + 0.1))
    #     maxes["x"].append(points_map[i])
    #     maxes["y"].append(group["perc"].mean())
    #     print(group["perc"].mean())
    #
    #     #plt.savefig(f"./images/error_dist_{points_map[i]}_points.png")
    #     #clear_figures()
    # #plt.savefig(f"./images/error_dist_all.png")
    # clear_figures()
    # ax = sns.barplot(x=maxes["x"], y=maxes["y"])
    # for i, (x, y) in enumerate(zip(maxes["x"], maxes["y"])):
    #     ax.text(x, y, y)
    # plt.savefig("./images/mean_error.png")
