//
// Created by Francesco Sgherzi on 7/2/20.
//

#ifndef CUDA_LLOC_H
#define CUDA_LLOC_H

#include "../utils/types.h"
#include "../utils/format_dataset.h"

EdgeList project_dataset_to_point(const dataset_t& dataset, const unsigned int point){
    EdgeList edgeList;
    for (auto triplet: dataset){
        if(triplet.i == point)
            edgeList.push_back({triplet.j, triplet.k});
    }

    return edgeList;

}

matrix_t to_adjacency_matrix(EdgeList& edgeList, const int n_points) {
    matrix_t adjacency_matrix;
    initialize_matrix(adjacency_matrix, (float) 0, n_points, n_points);

    for(auto edge: edgeList){
        adjacency_matrix[edge.u][edge.v] = 1;
        adjacency_matrix[edge.v][edge.u] = 1;
    }

    return adjacency_matrix;

}

embedding_t lloc(const dataset_t &dataset, const float epsilon, const unsigned int n_points, const int n_threads = (MULTITHREAD == true ? 8 : 1),
                 const unsigned int thread_id = 0) {


    /*
     * TODO:
     *  * Feedback arc set
     *  * Topological sort
     *  * WLLOC
     *  * Exhaustive enumeration
     *
     */

    if (DEBUG) {
        std::cout << "[TID " << thread_id << "]: " << "Running LLOC on dataset with " << dataset.size() << " triplet and epsilon " << epsilon << " on "
                  << n_threads << " threads " << std::endl;
    }

    /*
     * How do I iterate on all points in a multithreaded environment?
     *      chunk_size = n_points / n_threads;
     *      for i from thread_id * chunk_size until (thread_id + 1) * chunk_size
     */

    const unsigned int chunk_size = n_points / n_threads;
    for (unsigned int i = thread_id * chunk_size; i < (thread_id + 1) * chunk_size; ++i) {
        // Main loop
        EdgeList edgeList = project_dataset_to_point(dataset, i);
        matrix_t adjacency_matrix = to_adjacency_matrix(edgeList, n_points);

    }

    return embedding_t();
}

#endif //CUDA_LLOC_H
