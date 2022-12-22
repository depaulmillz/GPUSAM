#include <cassert>
#include <iostream>
#include <cstring>
#include <gpu/Linalg.cuh>
#include <tuple>
#include <chrono>
#include <unistd.h>

// Matricies are M by N

int main(int argc, char** argv) {

    int type = 0;

    char c;

    while((c = getopt(argc, argv, "p")) != -1) {
        switch (c) {
            case 'p':
                type = 1;
                break;
            case '?':
                std::cerr << "Unsure what arg " << c << " means" << std::endl; 
                return 1;
            default:
                std::cerr << "Unsure" << std::endl;
                return 2;
        }
    }

    const int size = 10000;

    setup();

    // A
    ColMajMatrix<HOST>* A = new ColMajMatrix<HOST>(size, size);
    
    A->zero();

    for(int i = 0; i < size; i++) {
        float f = rand();
        A->operator()(i, i) = f;
    }

    SparseMatrix<DEVICE> spA = A->copy<DEVICE>().toSparseWithNNZ<DEVICE>(size);

    delete A;

    // b
    Vector<HOST>* b = new Vector<HOST>(size);
    b->zero();
    b->operator[](0) = 1.0f;
    Vector<DEVICE> b_d = b->copy<DEVICE>();
    delete b;
    
    // x0
    Vector<DEVICE> x0{size};
    x0.zero();

    if(type == 1) {
        std::cerr << "Preconditioning" << std::endl;
    } else {
        std::cerr << "Base GMRES" << std::endl;
    }

    auto start = std::chrono::high_resolution_clock::now();
    Result res;
    if(type == 1) {
        SparseILU<DEVICE> M = ILU0<DEVICE>(spA);
        res = precondGmres(spA, b_d, x0, size, size - 1, M);
    } else {
        res = gmres<DEVICE>(spA, b_d, x0, size, size - 1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    int i = 0;
    std::cout << "iteration,error" << std::endl;
    for(auto& e : res.err) {
        std::cout << i << "," << e << std::endl;
        i++;
    }

    std::cout << std::chrono::duration<double>(end - start).count() << std::endl;

    cleanup();

    return 0;
}
