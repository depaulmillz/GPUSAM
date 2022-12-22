#include "Vector.cuh"
#include "DenseMatrix.cuh"
#include "SparseMatrix.cuh"
#include "SparseILU.cuh"
#include "GMRES.cuh"

#pragma once 

template<MemLoc M>
std::ostream& operator<<(std::ostream& os, const Vector<M>& v) {
    for(int i = 0; i < v.size; i++) {
        os << v[i] << "\n";
    }
    return os;
}

template<MemLoc M>
std::ostream& operator<<(std::ostream& os, const ColMajMatrix<M>& m) {
    for(int i = 0; i < m.rows; i++) {
        for(int j = 0; j < m.cols - 1; j++) {
            os << m(i, j) << " ";
        }
        os << m(i, m.cols - 1) << "\n";
    }
    return os;
}


