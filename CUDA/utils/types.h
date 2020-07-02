//
// Created by Francesco Sgherzi on 7/2/20.
//

#ifndef CUDA_TYPES_H

#define CUDA_TYPES_H
#include <vector>
#include <map>
#include <ostream>

struct Triplet {
    unsigned int i, j, k;
    friend std::ostream& operator<< (std::ostream &out, const Triplet& triplet){
        return out << "{" << triplet.i << ", " << triplet.j << ", " << triplet.k << "}";
    }
};

struct Arc {
    unsigned int u,v;
    friend std::ostream& operator<< (std::ostream &out, const Arc& arc){
        return out << "{" << arc.u << " -> " << arc.v << "}";
    }
};

typedef Triplet triplet_t;
typedef std::vector<triplet_t> dataset_t;

typedef std::vector<float> row_t;
typedef std::vector<row_t> matrix_t;
typedef std::map<int,float> embedding_t;
typedef std::vector<Arc> EdgeList;

#endif //CUDA_TYPES_H
