from os import environ

EPSILON                                = 1 / 15

MNIST_ROW_SIZE                         = 28
MNIST_COL_SIZE                         = 28
MNIST_SUBSAMPLE_FACTOR                 = 100 # UNUSED
MNIST_MEAN_VALUE_SCALE                 = 1.5 # UNUSED
MNIST_MIN_CORR_COEFF                   = 0.9 # UNUSED
MNIST_CONSTRAINT_INCLUSION_PROBABILITY = 1   # UNUSED
MNIST_DIGIT_EXCLUSION_PROBABILITY      = 0.0 # UNUSED
MNIST_ERROR_RATE                       = 0.0 # UNUSED
MNIST_BUCKETS_BASE_WIDTH               = 0.0 # UNUSED
USE_MULTIPROCESS                       = True
USE_DISTANCE                           = True
STE_NUM_DIGITS                         = 1000
CONTAMINATION_PERCENTAGE               = 0.5
SUPPORTED_DATASETS                     = ["mnist"]
GRAPH_MOCK                             = False # UNUSED
USE_MNIST                              = True
USE_RANDOM                             = False
TRAIN_TEST_SPLIT_RATE                  = 0.3
GRAPH_NUM_NODES                        = 4000 # UNUSED
ROE_SAMPLES                            = 1000
BAR_POSITION_OFFSET                    = 0

if environ.get("EPSILON"):
    EPSILON = float(environ.get("EPSILON"))

if environ.get("MNIST_DIGIT_EXCLUSION_PROBABILITY"):
    MNIST_DIGIT_EXCLUSION_PROBABILITY = float(environ.get("MNIST_DIGIT_EXCLUSION_PROBABILITY"))

if environ.get("MNIST_ERROR_RATE"):
    MNIST_ERROR_RATE = float(environ.get("MNIST_ERROR_RATE"))

if environ.get("USE_DISTANCE"):
    MNIST_ERROR_RATE = bool(int(environ.get("USE_DISTANCE")))

if environ.get("CONTAMINATION_PERCENTAGE"):
    CONTAMINATION_PERCENTAGE = float(environ.get("CONTAMINATION_PERCENTAGE"))

if environ.get("MNIST"):
    USE_MNIST = bool(int(environ.get("MNIST")))

if environ.get("RANDOM"):
    USE_RANDOM = bool(int(environ.get("MNIST")))

if environ.get("TRAIN_TEST_SPLIT_RATE"):
    TRAIN_TEST_SPLIT_RATE = float(environ.get("TRAIN_TEST_SPLIT_RATE"))