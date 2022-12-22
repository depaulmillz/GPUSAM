#include "Helpers.cuh"
#include "Vector.cuh"

#pragma once

template<MemLoc Mem>
struct ColMajMatrix {

    ColMajMatrix() : lda(0), rows(0), cols(0), values(nullptr) {}

    ColMajMatrix(int r, int c) : lda(r), rows(r), cols(c) {
        Alloc<Mem>{}(&values, lda * cols);
    }

    void zero() {
        Memset<Mem>{}(values, 0, sizeof(float) * lda * cols);   
    }

    ColMajMatrix(ColMajMatrix<Mem>&& other) : lda(other.lda), rows(other.rows), cols(other.cols), values(other.values) {
        other.values = nullptr;
    }

    ColMajMatrix(const ColMajMatrix&) = delete;

    ~ColMajMatrix() {
        Dealloc<Mem>{}(values);
    }

    Vector<Mem> col(int i) {
        Vector<Mem> v{};
        v.size = rows;
        v.values = values + i * lda;
        v.free = false;
        cusparseErrchk(cusparseCreateDnVec(&v.desc, v.size, v.values, CUDA_R_32F));
        return std::move(v);
    }

    const Vector<Mem> col(int i) const {
        Vector<Mem> v{};
        v.size = rows;
        v.values = values + i * lda;
        v.free = false;
        cusparseErrchk(cusparseCreateDnVec(&v.desc, v.size, v.values, CUDA_R_32F));
        return std::move(v);
    }

    Vector<Mem> copyCol(int i) const {
        Vector<Mem> v = const_cast<ColMajMatrix<Mem>*>(this)->col(i);
        auto v2 = v.copy();
        return std::move(v2);
    }

    template<MemLoc M>
    Vector<M> operator*(const Vector<M>& x) const {
        assert(cols == x.size);
        Vector<M> y{rows};
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const cublasOperation_t op = CUBLAS_OP_N;
        cublasErrchk(cublasSgemv(cublas_handle, op, rows, cols, &alpha, values, lda, x.values, 1, &beta, y.values, 1));
        gpuErrchk(cudaDeviceSynchronize());
        return y;
    }

    float& operator()(uint64_t i, uint64_t j) {
        assert(i < rows && j < cols);
        return values[j * lda + i];
    }

    const float& operator()(uint64_t i, uint64_t j) const {
        assert(i < rows && j < cols);
        return values[j * rows + i];
    }

    template<MemLoc M, std::enable_if_t<M != HOST, void*> = nullptr>
    SparseMatrix<M> toSparse() {
        // TODO weirdness here
        cusparseDnMatDescr_t dd = 0x0;
        int nnz = 0;
        float* values_cpy = nullptr;
        Alloc<HOST>{}(&values_cpy, lda * cols);
        Memcpy<Mem, HOST>{}(values_cpy, values, sizeof(float) * lda * cols); 
        for(int j = 0; j < cols; j++) {
            for(int i = 0; i < rows; i++) {
                if(reinterpret_cast<uint32_t&>(values_cpy[i + j * lda]) != 0) {
                    nnz++;
                }
            }
        }
        SparseMatrix<M> sp{rows, cols, nnz};
        size_t bufferSize;
        void* buffer;
        cusparseErrchk(cusparseCreateDnMat(&dd, rows, cols, lda, values, CUDA_R_32F, CUSPARSE_ORDER_COL));
        cusparseErrchk(cusparseDenseToSparse_bufferSize(cusparse_handle, dd, sp.desc, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize));
        gpuErrchk(cudaMalloc((void**)&buffer, bufferSize));
        cusparseErrchk(cusparseDenseToSparse_analysis(cusparse_handle, dd, sp.desc, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buffer));
        cusparseErrchk(cusparseDenseToSparse_convert(cusparse_handle, dd, sp.desc, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buffer));
        gpuErrchk(cudaDeviceSynchronize());
        cusparseErrchk(cusparseDestroyDnMat(dd));
        gpuErrchk(cudaFree(buffer));
        gpuErrchk(cudaPeekAtLastError());
        return std::move(sp);
    }

    template<MemLoc M, std::enable_if_t<Mem != HOST && M != HOST, void*> = nullptr>
    SparseMatrix<M> toSparseWithNNZ(int nnz) {
        // TODO weirdness here
        cusparseDnMatDescr_t dd;
        SparseMatrix<M> sp{rows, cols, nnz};
        size_t bufferSize;
        void* buffer;
        cusparseErrchk(cusparseCreateDnMat(&dd, rows, cols, lda, values, CUDA_R_32F, CUSPARSE_ORDER_COL));
        cusparseErrchk(cusparseDenseToSparse_bufferSize(cusparse_handle, dd, sp.desc, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize));
        gpuErrchk(cudaMalloc((void**)&buffer, bufferSize));
        cusparseErrchk(cusparseDenseToSparse_analysis(cusparse_handle, dd, sp.desc, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buffer));
        cusparseErrchk(cusparseDenseToSparse_convert(cusparse_handle, dd, sp.desc, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buffer));
        cusparseErrchk(cusparseDestroyDnMat(dd));
        gpuErrchk(cudaFree(buffer));
        return std::move(sp);
    }

    template<MemLoc M = Mem>
    ColMajMatrix<M> copy() const {
        ColMajMatrix<M> c{};
        c.lda = lda;
        c.rows = rows;
        c.cols = cols;
        Alloc<M>{}(&c.values, lda * cols);
        Memcpy<Mem, M>{}(c.values, values, sizeof(float) * lda * cols);
        return std::move(c);
    }

    template<MemLoc M>
    void setCol(int j, const Vector<M>& v) {
        assert(v.size == rows && j < cols);
        float* start = values + (j * rows);
        Memcpy<M, Mem>{}(start, v.values, sizeof(float) * v.size);
    }

    template<MemLoc M>
    void setCol(int j, Vector<M>&& v) {
        assert(v.size == rows && j < cols);
        float* start = values + (j * rows);
        Memcpy<M, Mem>{}(start, v.values, sizeof(float) * v.size);
    }

    template<MemLoc M>
    void setColSize(int j, int s, Vector<M>& v) {
        assert(v.size >= s);
        float* start = values + (j * rows);
        Memcpy<M, Mem>{}(start, v.values, sizeof(float) * s);
    }

    void truncateNumRows(int i) {
        rows = i;
    }
    
    void truncateNumCols(int j) {
        cols = j;
    }

    // Solve Ax = b with A upper triangular and set b to result
    template<MemLoc M, MemLoc Mem2 = Mem, std::enable_if_t<Mem2 == HOST, void*> = nullptr>
    void upperTriangularSolve(Vector<M>& b) {
        assert(cols == b.size);
        int k = cols - 1;
        ColMajMatrix<HOST> U_h{k + 1, rows};
        for(int j = 1; j <= rows; j++) {
            int m = k + 1 - j;
            for(int i = max(1, j - k); i <= j; i++) {
                U_h(m + i - 1, j - 1) = this->operator()(i - 1, j - 1);
            }
        }

        ColMajMatrix<M> U_d = U_h.copy<DEVICE>();

        cublasErrchk(cublasStbsv(cublas_handle, CUBLAS_FILL_MODE_UPPER,
                           CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                           rows, k, U_d.values, U_d.lda,
                           b.values, 1));
        gpuErrchk(cudaDeviceSynchronize());
    }
    
    template<MemLoc M>
    friend std::ostream& operator<<(std::ostream& os, const ColMajMatrix<M>& v);

    int lda;
    int rows;
    int cols;
    float* values;
};

