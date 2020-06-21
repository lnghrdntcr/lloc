from time import time
from time import sleep
from PIL import Image
from io import BytesIO
import requests as req
from multiprocessing import Process
from enum import Enum
from itertools import permutations

TRIPLET_TYPES = [
    "ONE_CLASS_TRIPLET",
    "TWO_CLASS_TRIPLET",
    "THREE_CLASS_TRIPLET"
]


class TripletType(Enum):
    ONE_CLASS_TRIPLET = "ONE_CLASS_TRIPLET"
    TWO_CLASS_TRIPLET = "TWO_CLASS_TRIPLET"
    THREE_CLASS_TRIPLET = "THREE_CLASS_TRIPLET"


def transform(dataset, type, download_images):
    """
    Transforms the urls of the datasets into indices of a map
    :param dataset: the full dataset
    :return: A dataset that has the same structure but with new references to the images as a map
    """
    cache = {}
    new_ds = []
    cur_idx = 0
    for row in dataset:
        row_split = row.split(",")
        # There is an image every 4 elements
        images = [str(row_split[0]), str(row_split[5]), str(row_split[10])]
        images = list(map(lambda x: x.replace("\"", ""), images))

        cur_idxs = []
        for base, image in enumerate(images):
            if not cache.get(image):
                # Get the image only if not in cache
                cache[image] = cur_idx
                cur_idx += 1
                if download_images:
                    download_image(base, cache, cur_idx, image, row_split, type)

            cur_idxs.append(cur_idx)

        if TripletType.ONE_CLASS_TRIPLET in row:
            # the 3 images share a common label
            for triplet in permutations(cur_idxs, 3):
                new_ds.append(triplet)

        elif TripletType.TWO_CLASS_TRIPLET in row:
            # Only 2 of the images share a common label
            pass
        elif TripletType.THREE_CLASS_TRIPLET in row:
            pass

    return cur_idx, cache


def download_image(base, cache, cur_idx, image, row_split, type):
    print(f"[{type.upper()}] Retrieving {image}...")
    image_resp = req.get(image)
    if image_resp.status_code == 200:
        img_file = Image.open(BytesIO(image_resp.content))
        h, w = img_file.size
        top_left_col = h * float(row_split[base * 5 + 1])
        bottom_right_col = h * float(row_split[base * 5 + 2])
        top_left_row = w * float(row_split[base * 5 + 3])
        bottom_right_row = w * float(row_split[base * 5 + 4])

        try:
            face = img_file.crop((top_left_col, top_left_row, bottom_right_col, bottom_right_row))
            face.save(f"./datasets/parsed/{type}/{cache[image]}.jpg")
        except Exception:
            print(f"[{type.upper()}] Error saving {image}")

    else:
        print(f"[{type.upper()}] Error retrieving {image}")


def read_and_crop(path, type):
    with open(path) as ds:
        begin_parsing = time()
        img_n, cache = transform(ds, type, True)
        print(f"Time for parsing {img_n} images -> {time() - begin_parsing}s")


if __name__ == "__main__":

    download_and_crop = False

    if download_and_crop:
        train_process = Process(target=read_and_crop,
                                args=("./datasets/FEC_dataset/faceexp-comparison-data-train-public.csv", "train"))
        test_process = Process(target=read_and_crop,
                               args=("./datasets/FEC_dataset/faceexp-comparison-data-test-public.csv", "test"))

        train_process.start()
        test_process.start()

        train_process.join()
        test_process.join()
