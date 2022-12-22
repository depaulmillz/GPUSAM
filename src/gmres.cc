#include <cpu/GMRES.h>

int main(int argc, char** argv) {

    const int size = 10000;

    Eigen::SparseMatrix<float> A{size,size};

    for(int i = 0; i < size; i++) {
        A.coeffRef(i, i) = rand();
    }

    Eigen::VectorXf b = A * Eigen::Matrix<float, size, 1>::Ones();
    Eigen::VectorXf x0 = Eigen::Matrix<float, size, 1>::Zero();

    Eigen::IncompleteLUT<float> M;
    M.compute(A);

    auto start = std::chrono::high_resolution_clock::now();

    auto err = gmresEigen(A, b, x0, size, size - 1, M);
    auto end = std::chrono::high_resolution_clock::now();
    int i = 0;
    std::cout << "iteration,error" << std::endl;
    for(auto& e : err) {
        std::cout << i << "," << e << std::endl;
        i++;
    }
    std::cout << std::chrono::duration<double>(end - start).count() << std::endl;
    return 0;
}
