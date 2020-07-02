#include <iostream>
#include "utils/types.h"
#include "utils/format_dataset.h"
#include "lloc/lloc.h"
/*
 * TODO:
 *     * Read CSVs with datasets [DONE]
 *     * Create datasets from distances
 *     * Create LLOC
 */
int main() {
    unsigned int n_points;
    dataset_t dataset = create_triplet_dataset("/home/francesco/Projects/lloc/datasets/iris/iris.csv", n_points);

    embedding_t embedding = lloc(dataset, 1.0 / 10, n_points);

    return 0;
}
