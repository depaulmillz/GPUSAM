#include <cassert>
#include <iostream>
#include <cstring>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <tuple>
#include <chrono>

// Matricies are M by N

std::pair<Eigen::VectorXf, Eigen::VectorXf> arnoldiEigen(Eigen::SparseMatrix<float> A, Eigen::MatrixXf Q, int k) { // for col 0, k == 0
    Eigen::VectorXf q = A * Q.col(k);


    Eigen::VectorXf h{k + 2, 1};

    for (int i = 0; i <= k; i++) {
        float h_i = q.dot(Q.col(i));
        q = q - h_i * Q.col(i);
        h.coeffRef(i, 0) = h_i;
        //q -= h[i] * Q.col(i);
    }


    //std::cerr << "h" << std::endl;
    //std::cerr << h << std::endl;

    float norm = q.lpNorm<2>();
    assert(h.rows() > k + 1);
    h.coeffRef(k+1, 0) = norm; 
    q /= norm;

    //std::cerr << "q" << std::endl;
    //std::cerr << q << std::endl << std::endl;
    
    //std::cerr << "h" << std::endl;
    //std::cerr << h << std::endl << std::endl;

    return {q, h};
}

std::pair<Eigen::VectorXf, Eigen::VectorXf> arnoldi_precondEigen(Eigen::SparseMatrix<float> A, Eigen::MatrixXf Q, int k, Eigen::IncompleteLUT<float>& M) { // for col 0, k == 0
    Eigen::VectorXf q = M.solve((A * Q.col(k)));

    Eigen::VectorXf h{k + 2, 1};

    for (int i = 0; i <= k; i++) {
        float h_i = q.dot(Q.col(i));
        q = q - h_i * Q.col(i);
        h.coeffRef(i, 0) = h_i;
        //q -= h[i] * Q.col(i);
    }


    //std::cerr << "h" << std::endl;
    //std::cerr << h << std::endl;

    float norm = q.lpNorm<2>();
    assert(h.rows() > k + 1);
    h.coeffRef(k+1, 0) = norm; 
    q /= norm;

    //std::cerr << "q" << std::endl;
    //std::cerr << q << std::endl << std::endl;
    
    //std::cerr << "h" << std::endl;
    //std::cerr << h << std::endl << std::endl;

    return {q, h};
}

std::pair<float, float> givens_rotationEigen(float a, float b) {
    float t = std::sqrt(a * a + b * b);
    float cs = a / t;
    float sn = b / t;
    return {cs, sn};
}

std::pair<float, float> apply_givensEigen(Eigen::MatrixXf& H, int colH, Eigen::VectorXf cs, Eigen::VectorXf sn, int k) {

    float cs_k, sn_k;

    for(int i = 0; i < k - 1; i++) {
        float temp = cs.coeff(i, 0) * H.coeff(i, colH) + sn.coeff(i, 0) * H.coeff(i + 1, colH);
        H.coeffRef(i+1,colH) = -sn.coeff(i, 0) * H.coeff(i, colH) + cs.coeff(i, 0) * H.coeff(i + 1, colH);
        H.coeffRef(i, colH) = temp;
    }
   
    auto p = givens_rotationEigen(H.coeff(k, colH), H.coeff(k + 1, colH));

    cs_k = p.first;
    sn_k = p.second;

    H.coeffRef(k, colH) = cs_k * H.coeffRef(k, colH) + sn_k * H.coeffRef(k + 1, colH);
    H.coeffRef(k + 1, colH) = 0;

    return {cs_k, sn_k};
}

inline void backsubEigen(const Eigen::MatrixXf& H, Eigen::VectorXf& y, int n) {
    for(int i = n; i >= 0; i--) {
        //std::cerr << "y_" << i << " /= H_" << i << " " << i << std::endl; 
        y.coeffRef(i, 0) /= H.coeff(i, i);
        for(int j = 0; j < i; j++) {
            //std::cerr << "y_" << j << " -= H_" << j << " " << i << " * y_" << i << std::endl; 
            y.coeffRef(j, 0) -= H.coeff(j, i) * y.coeffRef(i, 0);
        }
    }
}

std::vector<float> gmresEigen(Eigen::SparseMatrix<float> A, Eigen::VectorXf b, Eigen::VectorXf x0, int N, int iters, Eigen::IncompleteLUT<float>& M, float tol = 1e-8) {
    
    std::vector<float> err;

    M.compute(A);

    Eigen::VectorXf r = M.solve(b - A * x0);
    auto r_norm = r.lpNorm<2>();
    std::cout << r_norm << std::endl;

    float b_norm = b.lpNorm<2>();

    Eigen::VectorXf betaVec = Eigen::VectorXf::Zero(N, 1);

    
    betaVec.coeffRef(0,0) = r_norm;

    //std::cout << "Initial betaVec" << std::endl;
    //std::cout << betaVec << std::endl;


    err.push_back(r_norm/b_norm);

    //std::cout << "Initial error " << r_norm / b_norm << std::endl;

    Eigen::MatrixXf Q = r / r_norm;

    Eigen::MatrixXf H = Eigen::MatrixXf::Zero(iters + 1, 1);

    Eigen::VectorXf sn = Eigen::VectorXf::Zero(iters, 1);
    Eigen::VectorXf cs = Eigen::VectorXf::Zero(iters, 1);

    int end = iters - 1;

    for(int k = 0; k < iters; k++) {

        Q.conservativeResize(Eigen::NoChange, Q.cols() + 1);

        auto p = arnoldi_precondEigen(A, Q, k, M);

        assert(k + 1 < Q.cols());

        Q.col(k + 1) = p.first;
        assert(k < H.cols());
        for(int i = 0; i <= k+1; i++) {
            assert(i < H.rows());
            assert(k < H.cols());
            assert(i < p.second.rows());
            H.coeffRef(i, k) = p.second.coeffRef(i, 0);
        }

        //std::cout << "H post Arnoldi" << std::endl;
        //std::cout << H << std::endl;
        //std::cout << "Q post Arnoldi" << std::endl;
        //std::cout << Q << std::endl;

        auto p2 = apply_givensEigen(H, k, cs, sn, k);

        //std::cout << "H post givens" << std::endl;
        //std::cout << H << std::endl;

        cs.coeffRef(k, 0) = p2.first;
        sn.coeffRef(k, 0) = p2.second;

        //std::cout << "cs = " << p2.first << " sn = " << p2.second << std::endl;

        betaVec.coeffRef(k + 1, 0) = -p2.second * betaVec.coeffRef(k, 0);
        betaVec.coeffRef(k, 0) = p2.first * betaVec.coeffRef(k, 0);
        
        float error = abs(betaVec.coeffRef(k + 1, 0)) / b_norm;

        err.push_back(error);

        if(error <= tol) {
            //std::cout << "Ended at error " << error << std::endl;
            end = k;
            break;
        }

        if(k != iters - 1)
            H.conservativeResize(Eigen::NoChange, H.cols() + 1);
    }

    //std::cerr << "end =" << end << std::endl;

    Eigen::VectorXf y = betaVec(Eigen::seq(0, end), 0);

    //std::cout << "H" << std::endl << H(seq(0,end), seq(0,end)) << std::endl;
    //std::cout << "Q" << std::endl << Q << std::endl;
    //std::cout << "betaVec" << std::endl << y << std::endl;

    backsubEigen(H, y, end);
    
    //std::cout << "Hy" << std::endl;
    //std::cout << H(seq(0, end), seq(0, end)) * y << std::endl;

    auto x_m = x0 + Q(Eigen::all, Eigen::seq(0, end)) * y;

    //std::cout << "x_m" << std::endl;
    //std::cout << x_m << std::endl;

    return err;
}

