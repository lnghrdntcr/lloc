from os import environ

EPSILON                                = 1 / 9

MNIST_ROW_SIZE                         = 28
MNIST_COL_SIZE                         = 28
MNIST_SUBSAMPLE_FACTOR                 = 100
MNIST_MEAN_VALUE_SCALE                 = 1.5
MNIST_MIN_CORR_COEFF                   = 0.9
MNIST_CONSTRAINT_INCLUSION_PROBABILITY = 1
MNIST_DIGIT_EXCLUSION_PROBABILITY      = 0.0
MNIST_ERROR_RATE                       = 0.0
MNIST_BUCKETS_BASE_WIDTH               = 0.0
USE_MULTIPROCESS                       = True
USE_DISTANCE                           = True
SUPPORTED_DATASETS                     = ["mnist"]
GRAPH_MOCK                             = False
GRAPH_NUM_NODES                        = 4000

if ENVIRON_EPSILON := environ.get("EPSILON"):
    EPSILON = float(ENVIRON_EPSILON)

if ENVIRON_MNIST_DIGIT_EXCLUSION_PROBABILITY := environ.get("MNIST_DIGIT_EXCLUSION_PROBABILITY"):
    MNIST_DIGIT_EXCLUSION_PROBABILITY = float(ENVIRON_MNIST_DIGIT_EXCLUSION_PROBABILITY)

if ENVIRON_MNIST_ERROR_RATE := environ.get("MNIST_ERROR_RATE"):
    MNIST_ERROR_RATE = float(ENVIRON_MNIST_ERROR_RATE)

if ENVIRON_USE_DISTANCE := environ.get("USE_DISTANCE"):
    MNIST_ERROR_RATE = bool(int(ENVIRON_USE_DISTANCE))

