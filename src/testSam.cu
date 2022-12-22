#include <cassert>
#include <iostream>
#include <cstring>
#include <gpu/Linalg.cuh>
#include <tuple>
#include <chrono>
#include <unistd.h>

// Matricies are M by N

int main(int argc, char** argv) {

    int size = 5;

    setup();

    // A
    SparseMatrix<HOST>* A = new SparseMatrix<HOST>(size, size);
    SparseMatrix<HOST> A0{size, size};

    A->operator()(0, 0) = 2.0f;
    A->operator()(0, 1) = -1.0f;

    A0(0,0) = 2.0f;
    A0(0,1) = -1.0f;

    for(int i = 1; i < size - 1; i++) {        
        A->operator()(i, i) = 2.0f;
        A->operator()(i + 1, i) = -1.0f;
        A->operator()(i, i + 1) = -1.0f;
        A0(i,i) = 2.0f;
        A0(i + 1, i) = -1.0f;
        A0(i, i + 1) = -1.0f;
    }

    A->operator()(size - 1, size - 1) = 2.0f;
    A0(size - 1, size - 1) = 2.0f;

    SparseMatrix<DEVICE> spA1 = A->copy<DEVICE>();

    for(int i = 0; i < size; i++) {
        A->operator()(i,i) += 1.0f;
    }

    SparseMatrix<DEVICE> spA2 = A->copy<DEVICE>();
   

    delete A;

    SparseMatrix<DEVICE> N = sparseApproximateMap(spA2, A0);    

    cleanup();

    
    return 0;
}
