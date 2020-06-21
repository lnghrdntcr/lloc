from os import system
from tqdm import tqdm
from time import time

if __name__ == "__main__":
    for ds in tqdm(["RANDOM", "MNIST", "RANDOM"], desc="Dataset: ", position=0):
        USE_MNIST = int(ds == "MNIST")
        USE_RANDOM = int(ds == "RANDOM")
        for train_test_split_rate in tqdm([0.3, 0.5], desc="Train test split rate: ", position=1):
            for epsilon_values in tqdm([1 / i for i in range(8, 10)], desc="Epsilon: ", position=2):
                for p in tqdm([0, 0.1, 0.15, 0.20, 0.25], desc="Percentage: ", position=3):
                    for _ in range(5): 
                        for USE_ADDITIVE_WEIGHTS in [1, 0]:
                            begin = time()
                            system(
                                f"EPSILON={epsilon_values} ADDITIVE_WEIGHTS={USE_ADDITIVE_WEIGHTS} TRAIN_TEST_SPLIT_RATE={train_test_split_rate} CONTAMINATION_PERCENTAGE={p} MNIST={USE_MNIST} RANDOM={USE_RANDOM} python main.py >> results.csv")
                            print(f"Time to run = {time() - begin}s")
