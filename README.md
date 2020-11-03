# Learning Lines with Ordinal Constraints (LLOC)
An implementation of the line embedding algorithm from the paper [Learning Lines with Ordinal Constraints](https://drops.dagstuhl.de/opus/volltexte/2020/12648/) - [ArXiv](https://arxiv.org/abs/2004.13202)

## Installation
All dependencies with relative version are specified in `requirements.txt`, so it suffices to run ```pip install -r requirements.txt```

## Usage
```
from lloc import lloc
dataset = create_dataset() # created by end user
constraints, num_points = create_constraints() # created by end user, see Triplet Build for examples
embedding = lloc(constraints, num_points, dataset)
```

The `lloc` function creates an embedding starting from a  constraint set containing triplets in the form (a, b, c).

`lloc` is build to facilitate vertical parallelization via partitioning.
An helper function (`format_arguments`) is provided to partition the input pointset and the relative constraints.
See `main.py` for an example.

## Triplet build
`lloc` supports triplet constraints in the form (a, b, c). The algorithm is agnostic to how the triplet is build, since it is mostly domain dependent.
Examples on how to build the triplet constraints are provided in the `format_dataset/format_dataset.py` file (see `format_mnist_from_distances`, or `create_sine_dataset` functions).

## Authors
* Francesco Sgherzi 
* Diego Ihara
* Bohan Fan
* Anastasios Sidiropoulos
* Mina Valizadeh
* Neshat Mohammadi

## Acknowledgments
This work was supported by the National Science Foundation under award CAREER 1453472, and grants CCF-1815145, CCF-1934915.