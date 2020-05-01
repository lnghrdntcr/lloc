from os import system

if __name__ == "__main__":
    for epsilon_values in [1 / buckets for buckets in range(2, 10)]:
        system("clear")
        system(f"EPSILON={epsilon_values} python main.py >> results.txt")
        system("echo '\n\n\n' >> results.txt")
