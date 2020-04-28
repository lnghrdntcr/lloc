from os import system

for epsilon_values in [1 / buckets for buckets in range(2, 10)]:
    for error_rate_values in [i / 10 for i in range(3)]:
        system("clear")
        system(f"EPSILON={epsilon_values} MNIST_ERROR_RATE={error_rate_values} MNIST_DIGIT_EXCLUSION_PROBABILITY={0.0} python main.py")
