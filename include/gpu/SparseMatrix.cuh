#include <cublas_v2.h>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>
#include <cusparse.h>
#include <cusolverSp.h>
#include <iostream>
#include <type_traits>
#include <map>

#pragma once

template<MemLoc Mem>
struct SparseMatrix {

    explicit SparseMatrix(int r, int c, int n) : rows(r), cols(c), nnz(n), set(true), desc(0x0) {
        Alloc<Mem>{}(&rowPtr, rows + 1);
        Alloc<Mem>{}(&colIdx, nnz);
        Alloc<Mem>{}(&values, nnz);
        cusparseErrchk(cusparseCreateCsr(&this->desc, rows, cols, nnz, rowPtr, colIdx, values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    }

    SparseMatrix(const SparseMatrix<Mem>&) = delete;

    SparseMatrix(SparseMatrix<Mem>&& other) {
        set = other.set;
        rows = other.rows;
        cols = other.cols;
        nnz = other.nnz;
        desc = other.desc;
        rowPtr = other.rowPtr;
        colIdx = other.colIdx;
        values = other.values;
        other.set = false;
    }

    ~SparseMatrix() {
        if(set) {
            cusparseErrchk(cusparseDestroySpMat(desc));
            Dealloc<Mem>{}(rowPtr);
            Dealloc<Mem>{}(colIdx);
            Dealloc<Mem>{}(values);
        }
    }
   
    template<MemLoc M, std::enable_if_t<M != HOST, void*> = nullptr>
    Vector<M> operator*(const Vector<M>& x) const {
        assert(x.values != nullptr);
        Vector<M> y{rows};
        assert(cols == x.size);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        size_t bufferSize;

        cusparseErrchk(cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, desc, x.desc, &beta, y.desc, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
        gpuErrchk(cudaDeviceSynchronize());
        void* buffer;
        gpuErrchk(cudaMalloc(&buffer, bufferSize));
        cusparseErrchk(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, desc, x.desc, &beta, y.desc, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaFree(buffer));

        return std::move(y);

    }

    template<MemLoc M>
    Vector<M> asyncMult(const Vector<M>& x, void*& buffer) const {
        Vector<M> y{rows};
        assert(cols == x.size);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        size_t bufferSize;

        cusparseErrchk(cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, desc, x.desc, &beta, y.desc, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
        gpuErrchk(cudaMalloc(&buffer, bufferSize));
        cusparseErrchk(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, desc, x.desc, &beta, y.desc, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));

        return std::move(y);

    }

    ColMajMatrix<Mem> toDense() const {
        ColMajMatrix<Mem> A{rows, cols};
        cusparseDnMatDescr_t matA = 0x0;
        cusparseErrchk(cusparseCreateDnMat(&matA, rows, cols, A.lda, A.values, CUDA_R_32F, CUSPARSE_ORDER_COL));
        size_t size;
        char* buf;
        cusparseErrchk(cusparseSparseToDense_bufferSize(cusparse_handle, desc, matA, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &size));
        Alloc<DEVICE>{}(&buf, size);
        cusparseErrchk(cusparseSparseToDense(cusparse_handle, desc, matA, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, buf));
        gpuErrchk(cudaDeviceSynchronize());
        cusparseErrchk(cusparseDestroyDnMat(matA));
        Dealloc<DEVICE>{}(buf);
        return std::move(A);
    }

    template<MemLoc M, std::enable_if_t<M != HOST && Mem != HOST, void*> = nullptr>
    ColMajMatrix<M> operator*(const ColMajMatrix<M>& A) const {
        ColMajMatrix<M> C{rows, A.cols};
        cusparseDnMatDescr_t matA{0x0}, matC{0x0};
        cusparseErrchk(cusparseCreateDnMat(&matA, A.rows, A.cols, A.lda, A.values, CUDA_R_32F, CUSPARSE_ORDER_COL));
        cusparseErrchk(cusparseCreateDnMat(&matC, C.rows, C.cols, C.lda, C.values, CUDA_R_32F, CUSPARSE_ORDER_COL));
        size_t size;
        char* buf;
        const auto nt = CUSPARSE_OPERATION_NON_TRANSPOSE;
        float alpha = 1.0f;
        float beta = 0.0f;
        cusparseErrchk(cusparseSpMM_bufferSize(cusparse_handle, nt, nt, &alpha, desc, matA, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &size));
        Alloc<DEVICE>{}(&buf, size);
        cusparseErrchk(cusparseSpMM_preprocess(cusparse_handle, nt, nt, &alpha, desc, matA, &beta, matC,CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buf));
        cusparseErrchk(cusparseSpMM(cusparse_handle, nt, nt, &alpha, desc, matA, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buf));
        gpuErrchk(cudaDeviceSynchronize());
        cusparseErrchk(cusparseDestroyDnMat(matA));
        cusparseErrchk(cusparseDestroyDnMat(matC));
        Dealloc<DEVICE>{}(buf);
        return std::move(C);
    }

    template<MemLoc M>
    ColMajMatrix<M> transposeTimes(const ColMajMatrix<M>& A) const {
        ColMajMatrix<M> C{rows, A.cols};
        cusparseDnMatDescr_t matA{0x0}, matC{0x0};
        cusparseErrchk(cusparseCreateDnMat(&matA, A.rows, A.cols, A.lda, A.values, CUDA_R_32F, CUSPARSE_ORDER_COL));
        cusparseErrchk(cusparseCreateDnMat(&matC, C.rows, C.cols, C.lda, C.values, CUDA_R_32F, CUSPARSE_ORDER_COL));
        size_t size;
        char* buf;
        const auto nt = CUSPARSE_OPERATION_NON_TRANSPOSE;
        float alpha = 1.0f;
        float beta = 0.0f;
        cusparseErrchk(cusparseSpMM_bufferSize(cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, nt, &alpha, desc, matA, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &size));
        Alloc<DEVICE>{}(&buf, size);
        cusparseErrchk(cusparseSpMM_preprocess(cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, nt, &alpha, desc, matA, &beta, matC,CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buf));
        cusparseErrchk(cusparseSpMM(cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, nt, &alpha, desc, matA, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buf));
        gpuErrchk(cudaDeviceSynchronize());
        cusparseErrchk(cusparseDestroyDnMat(matA));
        cusparseErrchk(cusparseDestroyDnMat(matC));
        Dealloc<DEVICE>{}(buf);
        return std::move(C);
    }
 
    template<MemLoc M>
    Vector<HOST> backsub(const Vector<M>& b) {
        // Solve This x = b
        ColMajMatrix<Mem> U_m = toDense();
        
        ColMajMatrix<HOST> U = std::move(U_m.template copy<MemLoc::HOST>());

        Vector<HOST> x = b.template copy<HOST>();

        for(int i = U.rows - 1; i >= 0; i--) {
            for(int j = i + 1; j < U.rows; j++) {
                x[i] -= U(i, j) * x[j];
            }
            x[i] /= U(i, i);
        }

        return std::move(x);
    }

    template<MemLoc M>
    ColMajMatrix<M> backsub(const ColMajMatrix<M>& B) {
        // solving This A = B

        ColMajMatrix<M> A{cols, B.cols};

        for(int i = 0; i < B.cols; i++) {
            A.template setCol<HOST>(i, backsub(B.col(i)));
        }

        return std::move(A);
    }


    template<MemLoc M, std::enable_if_t<M != HOST && Mem != HOST, void*> = nullptr>
    ColMajMatrix<M> fNormLeastSq(const ColMajMatrix<M>& B) const {
        ColMajMatrix<M> X{cols, B.cols};

        csrqrInfo_t info;
        void* pBuffer;

        cusolverErrchk(cusolverSpCreateCsrqrInfo(&info));

        size_t internalDataInBytes, workspaceInBytes;

        float* v;

        Alloc<Mem>{}(&v, nnz * B.cols);

        int nnzA = nnz * B.cols;
       
        cusparseMatDescr_t descrA;

        cusparseErrchk(cusparseCreateMatDescr(&descrA));
        cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO); // A is base-0
        cusparseErrchk(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
        std::cerr << "Here" << std::endl;

        cusolverErrchk(cusolverSpXcsrqrAnalysisBatched(cusolver_handle, rows, cols, nnzA, descrA, rowPtr, colIdx, info));
       
        gpuErrchk(cudaDeviceSynchronize());

        // analysis

        cusolverErrchk(cusolverSpScsrqrBufferInfoBatched(cusolver_handle, rows, cols, nnzA, descrA, v, rowPtr, colIdx, B.cols, info, &internalDataInBytes, &workspaceInBytes));
        gpuErrchk(cudaMalloc(&pBuffer, workspaceInBytes));
        std::cerr << "Batching" << std::endl;
        cusolverErrchk(cusolverSpScsrqrsvBatched(cusolver_handle, rows, cols, nnzA, descrA, v, rowPtr, colIdx, B.values, X.values, B.cols, info, pBuffer));
        std::cerr << "Done" << std::endl;
        cusolverErrchk(cusolverSpDestroyCsrqrInfo(info));
        gpuErrchk(cudaFree(pBuffer));
        cusparseErrchk(cusparseDestroyMatDescr(descrA));
        Dealloc<Mem>{}(v);
        return std::move(X);
    }


    int rows = 0;
    int cols = 0;
    int nnz = 0;
    cusparseSpMatDescr_t desc = 0x0;
    int32_t* rowPtr = nullptr;
    int32_t* colIdx = nullptr;
    float* values = nullptr;
    bool set;
};

struct COOIdxCompare {
    bool operator()(const std::pair<int, int>& left, const std::pair<int, int>& right) const {
        if(left.first < right.first)
            return true;
        else if(left.first == right.first && left.second < right.second)
            return true;
        return false;
    }
};

template<>
struct SparseMatrix<HOST> {

    SparseMatrix(int r, int c) : rows(r), cols(c), nnz(0) {
    }

    explicit SparseMatrix(int r, int c, int n) : rows(r), cols(c), nnz(0) {
    }


    SparseMatrix(const SparseMatrix<HOST>&) = delete;

    SparseMatrix(SparseMatrix<HOST>&& other) {
        set = other.set;
        rows = other.rows;
        cols = other.cols;
        nnz = other.nnz;
        mapping = std::move(other.mapping);
    }

    ~SparseMatrix() {
    }

    SparseMatrix<HOST>& operator=(SparseMatrix<HOST>&& other) {
        set = other.set;
        rows = other.rows;
        cols = other.cols;
        nnz = other.nnz;
        mapping = std::move(other.mapping);
        return *this;
    }

    float& operator()(int i, int j) {
        assert(i < rows && j < cols);
        return mapping[std::make_pair(i, j)];
    }

    template<MemLoc M, std::enable_if_t<M != HOST, void*> = nullptr>
    SparseMatrix<M> copy() const {
        std::vector<int> rowPtr;
        std::vector<int> colPtr;
        std::vector<float> values;

        for(auto& e : mapping) {
            if(e.second != 0.0f) {
                rowPtr.push_back(e.first.first);
                colPtr.push_back(e.first.second);
                values.push_back(e.second);
            }
        }

        // create our sparse Matrix

        int nnz = static_cast<int>(rowPtr.size());

        SparseMatrix<M> sp{rows, cols, nnz};



        int* csrRowPtr = new int[rows + 1];
        Memset<HOST>{}(csrRowPtr, 0, sizeof(int) * (rows + 1));

        // Idea from https://stackoverflow.com/questions/23583975/convert-coo-to-csr-format-in-c
        for (int i = 0; i < nnz; i++) {
            csrRowPtr[rowPtr[i] + 1]++;
        }
        for (int i = 0; i < rows; i++) {
            csrRowPtr[i + 1] += csrRowPtr[i];
        }

        Memcpy<HOST, DEVICE>{}(sp.rowPtr, csrRowPtr, sizeof(int) * (rows + 1));
        Memcpy<HOST, DEVICE>{}(sp.colIdx, colPtr.data(), sizeof(int) * nnz);
        Memcpy<HOST, DEVICE>{}(sp.values, values.data(), sizeof(float) * nnz);

        delete[] csrRowPtr;

        return std::move(sp);
    }

    ColMajMatrix<HOST> toDense() const {

        ColMajMatrix<HOST> d{rows, cols};
        d.zero();

        for(auto& e : mapping) {
            d(e.first.first, e.first.second) = e.second;    
        }

        return std::move(d);
    }

    Vector<DEVICE> col(int i) const {
        
        assert(i < cols);

        Vector<HOST> x{rows};

        for(auto& e : mapping) {
            if(e.first.second == i) {
                x[e.first.first] = e.second;
            }
        }

        return std::move(x.template copy<DEVICE>());
    }
    
    Vector<HOST> backsub(const Vector<HOST>& b) {
        // Solve This x = b
        ColMajMatrix<HOST> U = toDense();
        
        Vector<HOST> x = b.copy();

        for(int i = U.rows - 1; i >= 0; i--) {
            for(int j = i + 1; j < U.rows; j++) {
                x[i] -= U(i, j) * x[j];
            }
            x[i] /= U(i, i);
        }

        return std::move(x);
    }

    int rows;
    int cols;
    int nnz;
    std::map<std::pair<int, int>, float, COOIdxCompare> mapping;
    bool set;
};

