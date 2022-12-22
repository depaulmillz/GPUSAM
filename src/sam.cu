#include <cassert>
#include <iostream>
#include <cstring>
#include <cpu/GMRES.h>
#include <tuple>
#include <chrono>
#include <unistd.h>

// Matricies are M by N

int main(int argc, char** argv) {

    int type = 0;
    int sam = 0;

    char c;

    int size = 100;
    
    while((c = getopt(argc, argv, "psn:")) != -1) {
        switch (c) {
            case 'p':
                type = 1;
                break;
            case 's':
                sam = 1;
                break;
            case 'n':
                size = atoi(optarg);
                break;
            case '?':
                std::cerr << "Unsure what arg " << c << " means" << std::endl; 
                return 1;
            default:
                std::cerr << "Unsure" << std::endl;
                return 2;
        }
    }


    setup();

    // A
    Eigen::SparseMatrix<float> A{size, size};

    A.coeffRef(0, 0) = 2.0f;
    A.coeffRef(0, 1) = -1.0f;

    A0(0,0) = 2.0f;
    A0(0,1) = -1.0f;

    for(int i = 1; i < size - 1; i++) {        
        A.coeffRef(i, i) = 2.0f;
        A.coeffRef(i + 1, i) = -1.0f;
        A.coeffRef(i, i + 1) = -1.0f;
        A0(i,i) = 2.0f;
        A0(i + 1, i) = -1.0f;
        A0(i, i + 1) = -1.0f;
    }

    A.coeffRef(size - 1, size - 1) = 2.0f;
    A0(size - 1, size - 1) = 2.0f;

    SparseMatrix<DEVICE> spA1 = A->copy<DEVICE>();

    for(int i = 0; i < size; i++) {
        A.coeffRef(i,i) += 1.0f;
    }

    SparseMatrix<DEVICE> spA2 = A->copy<DEVICE>();
   

    delete A;

    // b
    Vector<HOST>* b = new Vector<HOST>(size);
    b->zero();
    b->operator[](0) = 1.0f;
    b->operator[](1) = 1.0f;
    b->operator[](2) = 1.0f;
    Vector<DEVICE> b_d = b->copy<DEVICE>();
    delete b;
    
    // x0
    Vector<DEVICE> x0{size};
    x0.zero();

    if(type == 1) {
        std::cerr << "Preconditioning";
        if(sam == 1) {
            std::cerr << " with SAM";
        }
        std::cerr << std::endl;
    } else {
        std::cerr << "Base GMRES" << std::endl;
    }

    SparseILU<DEVICE> M;

    auto start = std::chrono::high_resolution_clock::now();
    Result res1, res2;
    if(type == 1) {
        M = ILU0<DEVICE>(spA1);
        res1 = precondGmres(spA1, b_d, x0, size, 10, M);
    } else {
        res1 = gmres<DEVICE>(spA1, b_d, x0, size, 10);
    }

    if(type == 1) {
        if(sam == 1) {
            std::cerr << "Doing sam" << std::endl;
            SparseMatrix<DEVICE> N = sparseApproximateMap(spA2, A0);
            res2 = precondSamGmres(spA2, b_d, x0, size, size, M, N);
        } else {
            M = ILU0<DEVICE>(spA2);
            res2 = precondGmres(spA2, b_d, x0, size, size, M);
        }
    } else {
        res2 = gmres<DEVICE>(spA2, b_d, x0, size, size);
    }

    auto end = std::chrono::high_resolution_clock::now();
    
    int i = 0;
    std::cout << "iteration,error" << std::endl;
    for(auto& e : res1.err) {
        std::cout << i << "," << e << std::endl;
        i++;
    }

    i = 0;
    for(auto& e : res2.err) {
        std::cout << i << "," << e << std::endl;
        i++;
    }
    
    std::cout << std::chrono::duration<double>(end - start).count() << std::endl;

    cleanup();

    
    return 0;
}
