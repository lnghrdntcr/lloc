from random import random as rand
from enum import Enum


class TripletType(Enum):
    ONE_CLASS_TRIPLET = "ONE_CLASS_TRIPLET"
    TWO_CLASS_TRIPLET = "TWO_CLASS_TRIPLET"
    THREE_CLASS_TRIPLET = "THREE_CLASS_TRIPLET"


def get_ds(path, type, early_stop_count=200):
    with open(path) as ds:
        cache = {}
        reverse_cache = {}
        crop_map = {}
        next_idx = 0

        new_ds = []
        for line in ds:
            if rand() > 0.5:
                continue
            row_split = line.split(",")
            # There is an image every 4 elements
            images = [str(row_split[0]), str(row_split[5]), str(row_split[10])]

            lru_cache = {}
            for base, image in enumerate(images):
                if (index := cache.get(image)) is None:
                    cache[image] = next_idx
                    lru_cache[image] = next_idx
                    reverse_cache[str(next_idx)] = image
                    crop_map[str(next_idx)] = [
                        row_split[base * 5 + 1], # top_left_col
                        row_split[base * 5 + 2], # bottom_right_col
                        row_split[base * 5 + 3], # top_left_row
                        row_split[base * 5 + 4]  # bottom_right_row
                    ]
                    next_idx += 1
                if index is not None:
                    lru_cache[image] = index
                    reverse_cache[str(next_idx)] = image

            if TripletType.THREE_CLASS_TRIPLET in row_split:
                new_ds.append([cache[label] for label in reversed(images)])
            else:
                new_ds.append([cache[label] for label in images])

            if next_idx >= early_stop_count:
                break

        return new_ds, reverse_cache, crop_map
