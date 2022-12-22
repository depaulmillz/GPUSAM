#include "Arnoldi.cuh"
#include "GivensRot.cuh"
#include "Precondition.cuh"

#pragma once

struct Result {
    std::vector<float> err;
    Vector<HOST> x; 
};

template<MemLoc M>
Result gmres(SparseMatrix<DEVICE>& A, const Vector<M>& b, const Vector<M>& x0, int N, int iters, float tol = 1e-8) {
    
    std::vector<float> err;

    Vector<M> r = b - A * x0;
    auto r_norm = r.norm();

    float b_norm = b.norm();

    Vector<M> betaVec{iters + 1};

    betaVec[0] = r_norm;

    err.push_back(r_norm/b_norm);

    ColMajMatrix<HOST> Q{r.rows(), iters + 1};

    Vector<M> rNormed = r / r_norm;

    for(int i = 0; i < r.rows(); i++) {
        Q(i, 0) = rNormed[i];
    }

    ColMajMatrix<HOST> H{iters + 1, iters};

    Vector<M> sn{iters};
    Vector<M> cs{iters};

    int end = iters - 1;

    for(int k = 0; k < iters; k++) {

        auto Qdev = Q.copy<DEVICE>();

        auto p = arnoldi<M>(A, Qdev, k);

        for(int i = 0; i < r.rows(); i++) {
            Q(i, k + 1) = p.first[i];
        }
        
        for(int i = 0; i <= k+1; i++) {
            H(i, k) = p.second[i];
        }

        auto p2 = apply_givens(H, k, cs, sn, k);

        cs[k] = p2.first;
        sn[k] = p2.second;

        betaVec[k + 1] = -p2.second * betaVec[k];
        betaVec[k] = p2.first * betaVec[k];
        
        float error = abs(betaVec[k + 1]) / b_norm;

        err.push_back(error);

        if(error <= tol) {
            end = k;
            break;
        }
    }

    H.truncateNumRows(end + 1);
    H.truncateNumCols(end + 1);

    betaVec.truncate(end + 1);

    Vector<M> y = betaVec.copy();

    H.upperTriangularSolve(y);

    Q.truncateNumCols(end + 1);

    ColMajMatrix<DEVICE> Q_d = Q.copy<DEVICE>();

    Vector<DEVICE> x_m = x0 + Q_d * y;

    Vector<HOST> x_m_h = x_m.copy<HOST>();
    
    return Result{std::move(err), std::move(x_m_h)};

}

template<>
Result gmres<DEVICE>(SparseMatrix<DEVICE>& A, const Vector<DEVICE>& b, const Vector<DEVICE>& x0, int N, int iters, float tol) {
    
    std::vector<float> err;
    err.reserve(iters + 1);
    ColMajMatrix<DEVICE> Q{b.rows(), iters + 1};
    Vector<HOST> betaVec{iters + 1};
    ColMajMatrix<HOST> H{iters + 1, iters};
    Vector<HOST> sn{iters};
    Vector<HOST> cs{iters};
    
    int end = iters - 1;
     
    float b_norm = b.norm();
    
    Vector<DEVICE> r = b.copy<DEVICE>();
    
    void* buffer;

    Vector<DEVICE> tmp = A.asyncMult(x0, buffer);
    tmp.asyncDivEqual(-1);
    r.asyncPlusEqual(tmp);
    auto r_norm = r.norm();
    
    gpuErrchk(cudaFree(buffer));
    
    betaVec[0] = r_norm;
    
    err.push_back(r_norm/b_norm);

    Vector<DEVICE> rNormed = r / r_norm;

    Q.setCol(0, rNormed);
    
    for(int k = 0; k < iters; k++) {
        specialArnoldi(A, Q, H, k);

        auto p2 = apply_givens<HOST>(H, k, cs, sn, k);

        cs[k] = p2.first;
        sn[k] = p2.second;

        betaVec[k + 1] = -p2.second * betaVec[k];
        betaVec[k] = p2.first * betaVec[k];
        
        float error = abs(betaVec[k + 1]) / b_norm;

        err.push_back(error);

        if(error <= tol) {
            end = k;
            break;
        }
    }

    H.truncateNumRows(end + 1);
    H.truncateNumCols(end + 1);

    betaVec.truncate(end + 1);

    Vector<DEVICE> y = betaVec.copy<DEVICE>();

    H.upperTriangularSolve(y);
    
    Q.truncateNumCols(end + 1);

    Vector<DEVICE> x_m = x0 + Q * y;

    Vector<HOST> x_m_h = x_m.copy<HOST>();
    
    return Result{err, std::move(x_m_h)};
}

Result precondGmres(SparseMatrix<DEVICE>& A, const Vector<DEVICE>& b, const Vector<DEVICE>& x0, int N, int iters, const SparseILU<DEVICE>& M, float tol = 1e-8) {
   
    
    std::vector<float> err;
    err.reserve(iters + 1);
    ColMajMatrix<DEVICE> Q{b.rows(), iters + 1};
    Vector<HOST> betaVec{iters + 1};
    ColMajMatrix<HOST> H{iters + 1, iters};
    Vector<HOST> sn{iters};
    Vector<HOST> cs{iters};
    
    int end = iters - 1;
     
    float b_norm = b.norm();
    
    Vector<DEVICE> r = b.copy<DEVICE>();
    
    void* buffer;

    Vector<DEVICE> tmp = A.asyncMult(x0, buffer);
    tmp.asyncDivEqual(-1);
    r.asyncPlusEqual(tmp);
    M.applyInvSelf(r);
    auto r_norm = r.norm();
    
    gpuErrchk(cudaFree(buffer));
    
    betaVec[0] = r_norm;
    
    err.push_back(r_norm/b_norm);

    Vector<DEVICE> rNormed = r / r_norm;

    Q.setCol(0, rNormed);
    
    for(int k = 0; k < iters; k++) {
        precondArnoldi(A, Q, H, k, M);

        auto p2 = apply_givens<HOST>(H, k, cs, sn, k);

        cs[k] = p2.first;
        sn[k] = p2.second;

        betaVec[k + 1] = -p2.second * betaVec[k];
        betaVec[k] = p2.first * betaVec[k];
        
        float error = abs(betaVec[k + 1]) / b_norm;

        err.push_back(error);

        if(error <= tol) {
            end = k;
            break;
        }
    }

    H.truncateNumRows(end + 1);
    H.truncateNumCols(end + 1);

    betaVec.truncate(end + 1);

    Vector<DEVICE> y = betaVec.copy<DEVICE>();

    H.upperTriangularSolve(y);
    
    Q.truncateNumCols(end + 1);

    Vector<DEVICE> x_m = x0 + Q * y;

    Vector<HOST> x_m_h = x_m.copy<HOST>();
    
    return Result{err, std::move(x_m_h)};

}

Result precondSamGmres(SparseMatrix<DEVICE>& A, const Vector<DEVICE>& b, const Vector<DEVICE>& x0, int N, int iters, const SparseILU<DEVICE>& M, const SparseMatrix<DEVICE>& Nk, float tol = 1e-8) {
    
    std::vector<float> err;
    err.reserve(iters + 1);
    ColMajMatrix<DEVICE> Q{b.rows(), iters + 1};
    Vector<HOST> betaVec{iters + 1};
    ColMajMatrix<HOST> H{iters + 1, iters};
    Vector<HOST> sn{iters};
    Vector<HOST> cs{iters};
    
    int end = iters - 1;
     
    float b_norm = b.norm();
    
    Vector<DEVICE> r = b.copy<DEVICE>();
    
    void* buffer;

    Vector<DEVICE> tmp = A.asyncMult(x0, buffer);
    tmp.asyncDivEqual(-1);
    r.asyncPlusEqual(tmp);
    M.applyInvSelf(r);
    gpuErrchk(cudaDeviceSynchronize());
    Vector<DEVICE> rtmp = Nk * r;
    r = std::move(rtmp);
    auto r_norm = r.norm();
    
    gpuErrchk(cudaFree(buffer));
    
    betaVec[0] = r_norm;
    
    err.push_back(r_norm/b_norm);

    Vector<DEVICE> rNormed = r / r_norm;

    Q.setCol(0, rNormed);
    
    for(int k = 0; k < iters; k++) {
        precondSamArnoldi(A, Q, H, k, M, Nk);

        auto p2 = apply_givens<HOST>(H, k, cs, sn, k);

        cs[k] = p2.first;
        sn[k] = p2.second;

        betaVec[k + 1] = -p2.second * betaVec[k];
        betaVec[k] = p2.first * betaVec[k];
        
        float error = abs(betaVec[k + 1]) / b_norm;

        err.push_back(error);

        if(error <= tol) {
            end = k;
            break;
        }
    }

    H.truncateNumRows(end + 1);
    H.truncateNumCols(end + 1);

    betaVec.truncate(end + 1);

    Vector<DEVICE> y = betaVec.copy<DEVICE>();

    H.upperTriangularSolve(y);
    
    Q.truncateNumCols(end + 1);

    Vector<DEVICE> x_m = x0 + Q * y;

    Vector<HOST> x_m_h = x_m.copy<HOST>();
    
    return Result{err, std::move(x_m_h)};

}
