#include <gpu/Linalg.cuh>
#include <iostream>

int main(int argc, char** argv) {
    setup();
    ColMajMatrix<HOST> I{10,10};
    std::cerr << "Creating a" << std::endl;
    Vector<HOST> a{10};
    std::cerr << "Creating b" << std::endl;
    Vector<HOST> b{10};

    for(int i = 0; i < 10; i++) {
        a[i] = 1.0f;//rand();
        b[i] = 1.0f;//rand();
        I(i,i) = 1.0f;
    }

    std::cerr << "Creating c" << std::endl;
    Vector<DEVICE> c = a.copy<DEVICE>() + b.copy<DEVICE>();
    Vector<HOST> c_h = c.copy<HOST>();

    for(int i = 0; i < 10; i++) {
        if(c_h[i] != a[i] + b[i]) {
            std::cerr << "Err" << std::endl;
            std::cerr << a << std::endl;
            std::cerr << b << std::endl;
            std::cerr << c_h << std::endl;
            cleanup();
            return 1;
        }
    }

    std::cerr << "Creating c2" << std::endl;
    ColMajMatrix<DEVICE> Idev = I.copy<DEVICE>();
    Vector<DEVICE> c2 = Idev * c;
    Vector<HOST> c2_h = c.copy<HOST>();

    for(int i = 0; i < 10; i++) {
        if(c2_h[i] != c_h[i]) {
            std::cerr << "Err" << std::endl;
            std::cerr << I << std::endl;
            std::cerr << c_h << std::endl;
            std::cerr << c2_h << std::endl;
            cleanup();
            return 2;
        }
    }

    std::cerr << "Creating sparse I" << std::endl;
    SparseMatrix<DEVICE> spI = Idev.toSparse<DEVICE>();

    std::cerr << "Creating new c3" << std::endl;
    Vector<DEVICE> c3 = spI * c;
    std::cerr << "Got new c3" << std::endl;
    c2 = std::move(c3);
    c2_h = c.copy<HOST>();

    for(int i = 0; i < 10; i++) {
        if(c2_h[i] != c_h[i]) {
            std::cerr << "Err" << std::endl;
            std::cerr << I << std::endl;
            std::cerr << c_h << std::endl;
            std::cerr << c2_h << std::endl;
            cleanup();
            return 2;
        }
    }

    cleanup();
    return 0;
}
