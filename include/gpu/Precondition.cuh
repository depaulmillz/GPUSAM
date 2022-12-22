#pragma once

/*
template<MemLoc M>
SparseMatrix<M> batchSparseApproximateMap(const SparseMatrix<M>& Ak, const ColMajMatrix<M>& A0) {
    const float tol = 1e12;
    const int reorder = 0;
    assert(Ak.rows == Ak.cols);
    ColMajMatrix<M> Ndense{Ak.rows, Ak.rows};
    int sing;
    cusparseMatDescr_t desc;
    cusparseErrchk(cusparseCreateMatDescr(&desc));
    
    for(int i = 0; i < Ak.rows; i++) {
        cusolverErrchk(cusolverSpScsrlsvqr(cusolver_handle, Ak.rows, Ak.nnz, desc, Ak.values, Ak.rowPtr, Ak.colIdx, A0.col(i).values, tol, reorder, Ndense.col(i).values, &sing));
        gpuErrchk(cudaDeviceSynchronize());
    }
    
    cusparseErrchk(cusparseDestroyMatDescr(desc));
    SparseMatrix<M> sp = Ndense.toSparse();
    return std::move(sp);
}
*/

template<MemLoc M>
std::pair<SparseMatrix<M>, SparseMatrix<M>> qr(const ColMajMatrix<M>& A) {
    
    ColMajMatrix<M> Q = A.copy();

    for(int i = 0; i < A.cols; i++) {

        // Get w = Q^T A_*,i
        // Then set Q_*,i = sum -w Q_*,j

        for(int j = 0; j < i; j++) {

            // Perform for the first i cols, Q^T A_*,i

            float f = Q.col(j).dot(A.col(i)); // e_{j} dot a_{i} = R_{j,i}
            //R(j, i) = f;
            Q.col(i) -= Q.col(j) * f; // u -= e_{j} * (e_{j} dot a_{i})
        }

        Q.col(i) /= Q.col(i).norm();

       // R(i, i) = A.col(i).dot(Q.col(i)); // R_{i, i}
    }

    // Q R = A, R = Q^T A
    SparseMatrix<M> q = Q.toSparse();

    ColMajMatrix<M> R = q.transposeTimes(A);

    SparseMatrix<M> r = R.toSparse();

    return {std::move(q), std::move(r)};
}


/*
template<MemLoc M>
SparseMatrix<M> sparseApproximateMap(const SparseMatrix<M>& Ak, const ColMajMatrix<M>& A0) {
    assert(Ak.rows == Ak.cols);
    ColMajMatrix<M> Ndense = Ak.fNormLeastSq(A0);
    SparseMatrix<M> sp = Ndense.toSparse();
    return std::move(sp);
}

template<MemLoc M>
SparseMatrix<M> sparseApproximateMap(const SparseMatrix<M>& Ak, const SparseMatrix<HOST>& A0) {
    static_assert(M != HOST, "Cannot use with host");
    assert(Ak.rows == Ak.cols);
    ColMajMatrix<M> Ndense = Ak.fNormLeastSq(A0.toDense().template copy<DEVICE>());
    SparseMatrix<M> sp = Ndense.toSparse();
    return std::move(sp);
}
*/

template<MemLoc M>
SparseMatrix<M> sparseApproximateMap(const SparseMatrix<M>& Ak, const ColMajMatrix<M>& A0) {
    const float tol = 1e12;
    const int reorder = 0;
    assert(Ak.rows == Ak.cols);
    ColMajMatrix<M> Ndense{Ak.rows, Ak.rows};

    int sing;
    cusparseMatDescr_t desc;
    cusparseErrchk(cusparseCreateMatDescr(&desc));
    cusparseErrchk(cusparseSetMatType(desc, CUSPARSE_MATRIX_TYPE_GENERAL));
    cusparseErrchk(cusparseSetMatIndexBase(desc, CUSPARSE_INDEX_BASE_ZERO));

    for(int i = 0; i < Ak.rows; i++) {
        cusolverErrchk(cusolverSpScsrlsvqr(cusolver_handle, Ak.rows, Ak.nnz, desc, Ak.values, Ak.rowPtr, Ak.colIdx, A0.col(i).values, tol, reorder, Ndense.col(i).values, &sing));
        gpuErrchk(cudaDeviceSynchronize());
    }
    
    cusparseErrchk(cusparseDestroyMatDescr(desc));
    SparseMatrix<M> sp = Ndense.toSparse();
    return std::move(sp);
}

template<MemLoc M>
SparseMatrix<M> sparseApproximateMap(const SparseMatrix<M>& Ak, const SparseMatrix<HOST>& A0_) {
    static_assert(M != HOST, "Cannot use with host");
    const float tol = 1e12;
    const int reorder = 0;
    assert(Ak.rows == Ak.cols);
    ColMajMatrix<M> Ndense{Ak.rows, Ak.rows};
    int sing;
    cusparseMatDescr_t desc;
    cusparseErrchk(cusparseCreateMatDescr(&desc));
   
    cusparseErrchk(cusparseSetMatType(desc, CUSPARSE_MATRIX_TYPE_GENERAL));
    cusparseErrchk(cusparseSetMatIndexBase(desc, CUSPARSE_INDEX_BASE_ZERO));

    ColMajMatrix<M> A0 = A0_.toDense().template copy<M>();


    for(int i = 0; i < Ak.rows; i++) {
        cusolverErrchk(cusolverSpScsrlsvqr(cusolver_handle, Ak.rows, Ak.nnz, desc, Ak.values, Ak.rowPtr, Ak.colIdx, A0.col(i).values, tol, reorder, Ndense.col(i).values, &sing));
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
    }


    cusparseErrchk(cusparseDestroyMatDescr(desc));
    SparseMatrix<M> sp = Ndense.template toSparse<M>();
    gpuErrchk(cudaPeekAtLastError());
    return std::move(sp);
}

template<MemLoc M>
SparseMatrix<M> sparseApproximateMapv2(const SparseMatrix<M>& Ak, const ColMajMatrix<M>& A0) {

    ColMajMatrix<M> Ak_d = Ak.toDense();

    std::cerr << "Starting qr" << std::endl;
    auto p = qr(Ak_d);
    std::cerr << "Ended qr" << std::endl;

    SparseMatrix<M> Q = std::move(p.first);
    SparseMatrix<M> R = std::move(p.second);
    
    // RN = Q^T A_0

    ColMajMatrix<M> RN = Q.transposeTimes(A0);    

    ColMajMatrix<M> NDense = R.backsub(RN);

    SparseMatrix<M> sp = NDense.toSparse();
    return std::move(sp);
}

template<MemLoc M>
SparseMatrix<M> sparseApproximateMapv2(const SparseMatrix<M>& Ak, const SparseMatrix<HOST>& A0) {
    static_assert(M != HOST, "Cannot use with host");
 
    ColMajMatrix<M> Ak_d = Ak.toDense();

    std::cerr << "Starting qr" << std::endl;
    auto p = qr(Ak_d);
    std::cerr << "Ending qr" << std::endl;

    SparseMatrix<M> Q = std::move(p.first);
    SparseMatrix<M> R = std::move(p.second);
    
    // RN = Q^T A_0

    ColMajMatrix<M> A0_m = A0.toDense().copy<M>();

    ColMajMatrix<M> RN = Q.transposeTimes(A0_m);    

    ColMajMatrix<M> NDense = R.backsub(RN);

    SparseMatrix<M> sp = NDense.toSparse();
    return std::move(sp);
}

template<MemLoc M>
SparseILU<M> ILU0(const SparseMatrix<M>& A) {
    csrilu02Info_t info;
    cusparseErrchk(cusparseCreateCsrilu02Info(&info));
    cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    assert(A.rows == A.cols);
    cusparseMatDescr_t desc;
    cusparseErrchk(cusparseCreateMatDescr(&desc));

    SparseILU<M> spILU = SparseILU<M>(A);

    int size;
    char* buf;
    cusparseErrchk(cusparseScsrilu02_bufferSize(cusparse_handle, A.rows, A.nnz, desc, A.values, A.rowPtr, A.colIdx, info, &size));
    Alloc<DEVICE>{}(&buf, size);

    cusparseErrchk(cusparseScsrilu02_analysis(cusparse_handle, spILU.rows, spILU.nnz, desc, spILU.values, spILU.rowPtr, spILU.colIdx, info, policy, buf));
    cusparseErrchk(cusparseScsrilu02(cusparse_handle, spILU.rows, spILU.nnz, desc, spILU.values, spILU.rowPtr, spILU.colIdx, info, policy, buf));

    gpuErrchk(cudaDeviceSynchronize());

    Dealloc<DEVICE>{}(buf);
    cusparseErrchk(cusparseDestroyMatDescr(desc));
    cusparseErrchk(cusparseDestroyCsrilu02Info(info));
    return std::move(spILU);
}


