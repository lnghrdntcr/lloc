from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from tqdm import tqdm

import os

from utils.read_dataset import load_datasets, load_poisoned

if __name__ == '__main__':

    base_dir = "../datasets/lloc_output/"
    N_NEIGHBORS = 5
    print("TESTING LLOC DATASETS")

    with open("../old_results/results_metric_learning.csv", "w+") as file:
        file.write(f"dataset,algorithm,n_neighbors,accuracy(%),type,dimensions\n")

    for file in sorted(os.listdir(base_dir)):

        if "csv" not in file:
            continue

        if "from-labels" in file:
            t = "labels"
        else:
            t = "features"

        fname = f"{base_dir}/{file}"

        print(f"Doing {fname}")
        knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
        dimensions = 1
        if "1d" in file:
            df = pd.read_csv(fname)
            df = df[["x", "label"]]

            x = df["x"].to_numpy().reshape(-1, 1)
            y = df["label"].to_numpy().reshape(-1, 1)
        else:
            dimensions = 2
            df = pd.read_csv(fname)
            x = df[["x", "y"]].to_numpy()
            y = df["label"].to_numpy()

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        knn.fit(x_train, y_train)
        errors = 0
        ans = knn.predict(x_test)
        for i in range(len(ans)):
            # print(f"{ans[i]} vs {y_test[i].ravel()[0]}")
            errors += int(ans[i] != y_test[i].ravel()[0])

        with open("../old_results/results_metric_learning.csv", "a+") as results_file:
            results_file.write(
                f"{file.split('-')[0]},lloc_knn,{N_NEIGHBORS},{(1 - errors / len(x_test)) * 100},{t},{dimensions}\n")

    print("TESTING ORIGINAL DATASETS")
    for x, y, dataset_name in tqdm(load_datasets()):
        print("Doing ", dataset_name)
        knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        knn.fit(x_train, y_train)
        errors = 0
        ans = knn.predict(x_test)
        for i in range(len(ans)):
            # print(f"{ans[i]} vs {y_test[i].ravel()[0]}")
            errors += int(ans[i] != y_test[i])

        with open("../old_results/results_metric_learning.csv", "a+") as results_file:
            results_file.write(f"{dataset_name},knn,{N_NEIGHBORS},{(1 - errors / len(x_test)) * 100},NA,NA\n")

    for x, y, dataset_name in tqdm(load_poisoned()):
        print("Doing ", dataset_name)

        knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        knn.fit(x_train, y_train)
        errors = 0
        ans = knn.predict(x_test)
        for i in range(len(ans)):
            # print(f"{ans[i]} vs {y_test[i].ravel()[0]}")
            errors += int(ans[i] != y_test[i])

        with open("../old_results/results_metric_learning.csv", "a+") as results_file:
            results_file.write(f"{dataset_name},knn,{N_NEIGHBORS},{(1 - errors / len(x_test)) * 100},NA,NA\n")

