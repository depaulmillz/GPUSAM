#include "SparseMatrix.cuh"

#pragma once

template<MemLoc Mem>
struct SparseILU {
    
    SparseILU() : rows(0), cols(0), nnz(0), buf(nullptr), set(false) {}

    explicit SparseILU(int r, int c, int n) : rows(r), cols(c), nnz(n), set(true), buf(nullptr), bufSize(0) {
        Alloc<Mem>{}(&rowPtr, rows + 1);
        Alloc<Mem>{}(&colIdx, nnz);
        Alloc<Mem>{}(&values, nnz);
    
        //cusparseErrchk(cusparseCreateMatDescr(&descL));
        //cusparseErrchk(cusparseSetMatIndexBase(descL, CUSPARSE_INDEX_BASE_ZERO));
        //cusparseErrchk(cusparseSetMatType(descL, CUSPARSE_MATRIX_TYPE_GENERAL));
        //cusparseErrchk(cusparseSetMatFillMode(descL, CUSPARSE_FILL_MODE_LOWER));
        //cusparseErrchk(cusparseSetMatDiagType(descL, CUSPARSE_DIAG_TYPE_UNIT));

        //cusparseErrchk(cusparseCreateMatDescr(&descU));
        //cusparseErrchk(cusparseSetMatIndexBase(descU, CUSPARSE_INDEX_BASE_ZERO));
        //cusparseErrchk(cusparseSetMatType(descU, CUSPARSE_MATRIX_TYPE_GENERAL));
        //cusparseErrchk(cusparseSetMatFillMode(descU, CUSPARSE_FILL_MODE_UPPER));
        //cusparseErrchk(cusparseSetMatDiagType(descU, CUSPARSE_DIAG_TYPE_NON_UNIT));

        cusparseErrchk(cusparseCreateCsr(&spDescL, rows, cols, nnz, rowPtr, colIdx, values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        cusparseErrchk(cusparseCreateCsr(&spDescU, rows, cols, nnz, rowPtr, colIdx, values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
        auto fillMode = CUSPARSE_FILL_MODE_LOWER;
        auto diagType = CUSPARSE_DIAG_TYPE_UNIT;
        cusparseErrchk(cusparseSpMatSetAttribute(spDescL, CUSPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode)));
        cusparseErrchk(cusparseSpMatSetAttribute(spDescL, CUSPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType)));
        
        fillMode = CUSPARSE_FILL_MODE_UPPER;
        diagType = CUSPARSE_DIAG_TYPE_NON_UNIT;
        cusparseErrchk(cusparseSpMatSetAttribute(spDescU, CUSPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode)));
        cusparseErrchk(cusparseSpMatSetAttribute(spDescU, CUSPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType)));
 
        //cusparseErrchk(cusparseCreateCsrsv2Info(&svL));
        //cusparseErrchk(cusparseCreateCsrsv2Info(&svU));
        //int size1, size2;
        //cusparseErrchk(cusparseScsrsv2_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, rows, nnz, descL, values, rowPtr, colIdx, svL, &size1));
        //cusparseErrchk(cusparseScsrsv2_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, rows, nnz, descU, values, rowPtr, colIdx, svU, &size2));
        cusparseErrchk(cusparseSpSV_createDescr(&svL));
        cusparseErrchk(cusparseSpSV_createDescr(&svU));

    }

    SparseILU(const SparseILU<Mem>&) = delete;

    explicit SparseILU(const SparseMatrix<Mem>& A) : SparseILU(A.rows, A.cols, A.nnz) {
        Memcpy<Mem, Mem>{}(rowPtr, A.rowPtr, sizeof(int32_t) * (rows + 1));
        Memcpy<Mem, Mem>{}(colIdx, A.colIdx, sizeof(int32_t) * nnz);
        Memcpy<Mem, Mem>{}(values, A.values, sizeof(float) * nnz);
    }

    SparseILU(SparseILU<Mem>&& other) {
        set = other.set;
        rows = other.rows;
        cols = other.cols;
        nnz = other.nnz;
        //descL = other.descL;
        spDescL = other.spDescL;
        //descU = other.descU;
        spDescU = other.spDescU;
        rowPtr = other.rowPtr;
        colIdx = other.colIdx;
        values = other.values;
        svL = other.svL;
        svU = other.svU;
        buf = other.buf;
        bufSize = other.bufSize;
        other.set = false;
    }

    ~SparseILU() {
        if(set) {
            cusparseErrchk(cusparseDestroySpMat(spDescL));
            cusparseErrchk(cusparseDestroySpMat(spDescU));
            Dealloc<Mem>{}(rowPtr);
            Dealloc<Mem>{}(colIdx);
            Dealloc<Mem>{}(values);
            cusparseErrchk(cusparseSpSV_destroyDescr(svL));
            cusparseErrchk(cusparseSpSV_destroyDescr(svU));
            gpuErrchk(cudaFree(buf));
        }
    }

    SparseILU<Mem>& operator=(SparseILU<Mem>&& other) {
        if(set) {
            cusparseErrchk(cusparseDestroySpMat(spDescL));
            cusparseErrchk(cusparseDestroySpMat(spDescU));
            Dealloc<Mem>{}(rowPtr);
            Dealloc<Mem>{}(colIdx);
            Dealloc<Mem>{}(values);
            cusparseErrchk(cusparseSpSV_destroyDescr(svL));
            cusparseErrchk(cusparseSpSV_destroyDescr(svU));
            gpuErrchk(cudaFree(buf));
        }

        set = other.set;
        rows = other.rows;
        cols = other.cols;
        nnz = other.nnz;
        //descL = other.descL;
        spDescL = other.spDescL;
        //descU = other.descU;
        spDescU = other.spDescU;
        rowPtr = other.rowPtr;
        colIdx = other.colIdx;
        values = other.values;
        svL = other.svL;
        svU = other.svU;
        buf = other.buf;
        bufSize = other.bufSize;
        other.set = false;
        return *this;
    }

    // U^{-1}L^{-1}x
    template<MemLoc M>
    Vector<M> applyInv(const Vector<M>& x) const {
        // LU y = x
        // Lz = x
        // Uy = z
        Vector<M> y{x.size};
        Vector<M> z{x.size};
        float alpha = 1.0f;

        size_t size1, size2;

        cusparseErrchk(cusparseSpSV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spDescL, x.desc, z.desc, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, svL, &size1));
        cusparseErrchk(cusparseSpSV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spDescU, z.desc, y.desc, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, svU, &size2));

        if(std::max(size1, size2) > bufSize) {
            if(buf != nullptr) gpuErrchk(cudaFree(buf));
            const_cast<SparseILU<Mem>*>(this)->bufSize = std::max(size1, size2);
            gpuErrchk(cudaMalloc(&const_cast<SparseILU<Mem>*>(this)->buf, std::max(size1, size2)));
        }
        
        cusparseErrchk(cusparseSpSV_analysis(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spDescL, x.desc, z.desc, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, svL, buf));
        cusparseErrchk(cusparseSpSV_analysis(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spDescU, z.desc, y.desc, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, svU, buf));
        cusparseErrchk(cusparseSpSV_solve(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spDescL, x.desc, z.desc, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, svL));
        cusparseErrchk(cusparseSpSV_solve(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spDescU, z.desc, y.desc, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, svU));
        gpuErrchk(cudaDeviceSynchronize());
        return std::move(y);
    }

    template<MemLoc M, std::enable_if_t<M != HOST, void*> = nullptr>
    Vector<M>& applyInvSelf(Vector<M>& x) const {
        // LU y = x
        // Lz = x
        // Uy = z
        Vector<M> z{x.size};
        float alpha = 1.0f;

        size_t size1, size2;

        cusparseErrchk(cusparseSpSV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spDescL, x.desc, z.desc, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, svL, &size1));
        cusparseErrchk(cusparseSpSV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spDescU, z.desc, x.desc, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, svU, &size2));
        
        if(std::max(size1, size2) > bufSize) {
            if(buf != nullptr) gpuErrchk(cudaFree(buf));
            const_cast<SparseILU<Mem>*>(this)->bufSize = std::max(size1, size2);
            gpuErrchk(cudaMalloc(&const_cast<SparseILU<Mem>*>(this)->buf, std::max(size1, size2)));
        }

        cusparseErrchk(cusparseSpSV_analysis(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spDescL, x.desc, z.desc, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, svL, buf));
        cusparseErrchk(cusparseSpSV_analysis(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spDescU, z.desc, x.desc, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, svU, buf));
        cusparseErrchk(cusparseSpSV_solve(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spDescL, x.desc, z.desc, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, svL));
        cusparseErrchk(cusparseSpSV_solve(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spDescU, z.desc, x.desc, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, svU));
        gpuErrchk(cudaDeviceSynchronize());

        return x;
    }
    
    // LUx
    template<MemLoc M>
    Vector<M> operator*(const Vector<M>& x) const {
        Vector<M> y{rows};
        assert(cols == x.size);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        size_t size1, size2;

        cusparseErrchk(cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spDescU, x.desc, &beta, y.desc, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &size1));
        cusparseErrchk(cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spDescL, y.desc, &beta, y.desc, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &size2));

        if(std::max(size1, size2) > bufSize) {
            if(buf != nullptr) gpuErrchk(cudaFree(buf));
            const_cast<SparseILU<Mem>*>(this)->bufSize = std::max(size1, size2);
            gpuErrchk(cudaMalloc(&const_cast<SparseILU<Mem>*>(this)->buf, std::max(size1, size2)));
        }

        cusparseErrchk(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spDescU, x.desc, &beta, y.desc, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buf));

        // y = Ux

        // y = Ly
        cusparseErrchk(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spDescL, y.desc, &beta, y.desc, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buf));
        gpuErrchk(cudaDeviceSynchronize());

        return std::move(y);
    }

    int rows;
    int cols;
    int nnz;
    //cusparseMatDescr_t descL;
    cusparseSpMatDescr_t spDescL;
    //cusparseMatDescr_t descU;
    cusparseSpMatDescr_t spDescU;
    //csrsv2Info_t svL;
    //csrsv2Info_t svU;
    cusparseSpSVDescr_t svL;
    cusparseSpSVDescr_t svU;
    int32_t* rowPtr;
    int32_t* colIdx;
    float* values;
    bool set;
    void* buf;
    size_t bufSize;
};

