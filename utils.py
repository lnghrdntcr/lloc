from io import BytesIO
from math import factorial
from os import system, mkdir
import requests as req
from PIL import Image
from config import EPSILON


def save_results(embeddings, image_cache, crop_map):
    for image_index, directory in embeddings.items():
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
            face.save(f"./results/{directory}/{image_index}.jpg")


def n_choose_k(n, k):
    assert n > k
    num = factorial(n)
    den = factorial(k) * factorial(n - k)

    return num / den


def setup_results_directories():
    system("rm -rf results/*")
    for i in range(int(1 / EPSILON) + 1):
        mkdir(f"./results/{i}")


def pretty_print_embedding(embedding):
    pretty_print = ", ".join([f"{el} -> {idx + 1}" for idx, el in enumerate(embedding)])
    print(f"Current embedding into R^1 = [{i} -> 0, {pretty_print}]")