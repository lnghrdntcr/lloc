//
// Created by Francesco Sgherzi on 7/2/20.
//

#ifndef CUDA_FORMAT_DATASET_H
#define CUDA_FORMAT_DATASET_H

#include "types.h"
#include "config.h"
#include "rapidcsv.h"
#include <cmath>
#include <string>

matrix_t to_dataset_matrix(const rapidcsv::Document &doc) {
    matrix_t matrix;
    const int DIM_Y = doc.GetRowCount() - 1;
    for (int i = 0; i < DIM_Y; ++i) {
        matrix.push_back(doc.GetRow<float>(i));
    }

    return matrix;
}

float l2_norm(const row_t &a, const row_t &b) {
    assert(a.size() == b.size());
    float acc = 0;
    for (int i = 0; i < a.size(); ++i) {
        float tmp = a[i] - b[i];
        acc += tmp * tmp;
    }

    return std::sqrt(acc);
}

template<typename T>
void initialize_matrix(matrix_t &matrix, T with, const int DIM_X, const int DIM_Y) {

    for (int i = 0; i < DIM_Y; ++i) {
        std::vector<T> row(DIM_X, with);
        matrix.push_back(row);
    }

}

void print_matrix(const matrix_t &matrix, const std::string &fmt = std::string("\t%.2f"), const int x_lim = 10,
                  const int y_lim = 10) {

    for (int i = 0; i < matrix.size() && i < x_lim; ++i) {
        if (i == x_lim - 1 || i == matrix.size() - 1){
            std::cout << "\t⋮" << std::endl;
            break;
        }


        for (int j = 0; j < matrix[i].size() && j < y_lim; ++j) {
            printf(fmt.c_str(), matrix[i][j]);

            if (j == y_lim - 1 || j == matrix[i].size() - 1) {
                std::cout << " ...\n";
                break;
            }

        }
    }
}

dataset_t create_triplet_dataset(const std::string &path, unsigned int& n_points_ret) {
    matrix_t distance_matrix;
    dataset_t dataset;

    rapidcsv::Document document(path, rapidcsv::LabelParams(-1, -1));
    matrix_t matrix = to_dataset_matrix(document);
    const int N_POINTS = matrix.size();
    n_points_ret = N_POINTS;
    initialize_matrix<float>(distance_matrix, 0.0, N_POINTS, N_POINTS);

    if (DEBUG) {
        std::cout << "Read " << path << ".\nDimensions: " << document.GetRowCount() << " Rows x "
                  << document.GetColumnCount()
                  << " Columns" << std::endl;
    }

    // Creation of distance Matrix
    for (int i = 0; i < N_POINTS; ++i) {
        for (int j = 0; j < N_POINTS; ++j) {
            distance_matrix[i][j] = l2_norm(matrix[i], matrix[j]);
        }
    }

    if (DEBUG){
        std::cout << "Distance Matrix: " << std::endl;
        print_matrix(distance_matrix);
    }

    // Triplet Creation
    for (unsigned int i = 0; i < N_POINTS; ++i) {
        for (unsigned int j = 0; j < N_POINTS; ++j) {
            if (i != j) {
                for (unsigned int k = 0; k < N_POINTS; ++k) {
                    // For all distinct triplets
                    if (i != k && j != k) {

                        if (distance_matrix[i][j] < distance_matrix[i][k]) {
                            dataset.push_back({i, j, k});
                        }

                    }
                }

            }
        }
    }
    return dataset;
}

#endif //CUDA_FORMAT_DATASET_H
