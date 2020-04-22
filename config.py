EPSILON                                = 1 / 9
MNIST_ROW_SIZE                         = 28
MNIST_COL_SIZE                         = 28
MNIST_SUBSAMPLE_FACTOR                 = 40
MNIST_MEAN_VALUE_SCALE                 = 1.5
MNIST_MIN_CORR_COEFF                   = 0.9
MNIST_CONSTRAINT_INCLUSION_PROBABILITY = 1
MNIST_DIGIT_EXCLUSION_PROBABILITY      = 0.1
MNIST_ERROR_RATE                       = 0.0
USE_PAGERANK                           = False
SUPPORTED_DATASETS                     = ["mnist"]
GRAPH_MOCK                             = False
GRAPH_NUM_NODES                        = 4000

# I want the size of the interval from which I'm embedding the points to be inversly proportional to the class distribution
# for instance -> if all elements are distributed uniformly the buckets should be somewhat contiguos and the range is (0.5 - base, 0.5 + base)
# but if the presence of a class is half of the average I want to make overlaps possible -> for instance if the class of 1 is 0.5 of the size of the maximum class
# I want the interval to be twice as big!
# So it should be something like (0.5 * scale_factor - base, 0.5 * scale_factor + base)
