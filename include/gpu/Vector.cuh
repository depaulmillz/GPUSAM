#include "Helpers.cuh"

#pragma once

__global__ void normKern(float* res, float* values, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < size) {
        float value = values[tid];
        value *= value;
        atomicAdd(res, value);
    }
};

void myNorm(int size, float* values, float* res) {
    float* gpuRes;
    Alloc<DEVICE>{}(&gpuRes, 1);
    Memset<DEVICE>{}(gpuRes, 0, sizeof(float));
    normKern<<<(size + 127/128), 128>>>(gpuRes, values, size);
    gpuErrchk(cudaDeviceSynchronize());
    Memcpy<DEVICE, HOST>{}(res, gpuRes, sizeof(float));
    Dealloc<DEVICE>{}(gpuRes);
    *res = std::sqrt(*res);
}

template<MemLoc Mem>
struct Vector {

    Vector() : size(0), values(nullptr), desc(0x0) {}

    explicit Vector(int s) : size(s) {
        Alloc<Mem>{}(&values, size);
        cusparseErrchk(cusparseCreateDnVec(&desc, size, values, CUDA_R_32F));
    }

    void zero() {
        Memset<Mem>{}(values, 0, sizeof(float) * size);
    }

    Vector(const Vector<Mem>&) = delete;

    Vector(Vector<Mem>&& other) { 
        free = other.free;
        values = other.values;
        size = other.size;
        desc = other.desc;

        other.values = nullptr;
        other.free = false;
    }

    Vector<Mem>& operator=(Vector<Mem>&& other) {
        if(free) {
            Dealloc<Mem>{}(values);
            cusparseErrchk(cusparseDestroyDnVec(desc));
        }

        free = other.free;
        values = other.values;
        size = other.size;
        desc = other.desc;
        other.values = nullptr;
        other.free = false;

        return *this;
    }

    ~Vector() {
        if(free) {
            Dealloc<Mem>{}(values);
            cusparseErrchk(cusparseDestroyDnVec(desc));
        }
        values = nullptr;
    }

    template<MemLoc M, std::enable_if_t<Mem != HOST && M != HOST, void*> = nullptr>
    float dot(const Vector<M>& v) const {
        float res;
        assert(size == v.size);
        cublasErrchk(cublasSdot(cublas_handle, size, values, 1, v.values, 1, &res));
        gpuErrchk(cudaDeviceSynchronize());
        return res;
    }

    template<MemLoc M, std::enable_if_t<Mem != HOST && M != HOST, void*> = nullptr>
    void asyncDot(const Vector<M>& v, float* res) const {
        assert(size == v.size);
        cublasErrchk(cublasSdot(cublas_handle, size, values, 1, v.values, 1, res));
    }

    template<MemLoc M, std::enable_if_t<Mem != HOST && M != HOST, void*> = nullptr>
    Vector<Mem>& operator+=(const Vector<M>& v) {
        assert(size == v.size);
        const float alpha = 1.0f;
        // values = values + v.values;
        cublasErrchk(cublasSaxpy(cublas_handle, size, &alpha, v.values, 1, values, 1));
        gpuErrchk(cudaDeviceSynchronize());
        return *this;
    }

    template<MemLoc M, std::enable_if_t<Mem != HOST && M != HOST, void*> = nullptr>
    Vector<Mem>& asyncPlusEqual(const Vector<M>& v) {
        assert(size == v.size);
        const float alpha = 1.0f;
        // values = values + v.values;
        cublasErrchk(cublasSaxpy(cublas_handle, size, &alpha, v.values, 1, values, 1));
        return *this;
    }

    template<MemLoc M, std::enable_if_t<Mem != HOST && M != HOST, void*> = nullptr>
    Vector<Mem>& addScaled(float alpha, const Vector<M>& v) {
        assert(size == v.size);
        // values = values + alpha * v.values;
        cublasErrchk(cublasSaxpy(cublas_handle, size, &alpha, v.values, 1, values, 1));
        gpuErrchk(cudaDeviceSynchronize());
        return *this;
    }

    template<MemLoc M, std::enable_if_t<Mem != HOST && M != HOST, void*> = nullptr>
    void asyncAddScaled(float* alpha, const Vector<M>& v) {
        assert(size == v.size);
        // values = values + alpha * v.values;
        cublasErrchk(cublasSaxpy(cublas_handle, size, alpha, v.values, 1, values, 1));
    }
    
    template<MemLoc M, std::enable_if_t<Mem != HOST && M != HOST, void*> = nullptr>
    Vector<Mem>& operator-=(const Vector<M>& v) {
        assert(size == v.size);
        const float alpha = -1.0f;
        // values = values + v.values;
        cublasErrchk(cublasSaxpy(cublas_handle, size, &alpha, v.values, 1, values, 1));
        gpuErrchk(cudaDeviceSynchronize());
        return *this;
    }

    template<MemLoc M, std::enable_if_t<Mem != HOST && M != HOST, void*> = nullptr>
    Vector<Mem> operator+(const Vector<M>& v) const {
        auto c = this->copy<Mem>();
        c += v;
        return std::move(c);
    }

    template<MemLoc M, std::enable_if_t<Mem != HOST && M != HOST, void*> = nullptr>
    Vector<Mem> operator-(const Vector<M>& v) const {
        Vector<Mem> c = this->copy<Mem>();
        c -= v;
        return std::move(c);
    }

    template<MemLoc M = Mem, std::enable_if_t<M != HOST, void*> = nullptr>
    Vector<Mem>& operator/=(float alpha) {
        const float a = 1/alpha;
        cublasErrchk(cublasSscal(cublas_handle, size, &a, values, 1));
        gpuErrchk(cudaDeviceSynchronize());
        return *this;
    }

    template<MemLoc M = Mem, std::enable_if_t<M != HOST, void*> = nullptr>
    Vector<Mem>& operator*=(float alpha) {
        cublasErrchk(cublasSscal(cublas_handle, size, &alpha, values, 1));
        gpuErrchk(cudaDeviceSynchronize());
        return *this;
    }
    
    template<MemLoc M = Mem, std::enable_if_t<M != HOST, void*> = nullptr>
    Vector<Mem>& asyncDivEqual(float alpha) {
        const float a = 1/alpha;
        cublasErrchk(cublasSscal(cublas_handle, size, &a, values, 1));
        return *this;
    }

    template<MemLoc M = Mem, std::enable_if_t<M != HOST, void*> = nullptr>
    Vector<Mem> operator/(float alpha) const {
        Vector<Mem> c = copy<Mem>();
        c /= alpha;
        return std::move(c);
    }

    template<MemLoc M = Mem, std::enable_if_t<M != HOST, void*> = nullptr>
    Vector<Mem> operator*(float alpha) const {
        Vector<Mem> c = copy<Mem>();
        c *= alpha;
        return std::move(c);
    }

    template<MemLoc M = Mem, std::enable_if_t<M != DEVICE, void*> = nullptr>
    float& operator[](int i) {
        return this->values[i];
    }

    template<MemLoc M = Mem, std::enable_if_t<M != DEVICE, void*> = nullptr>
    const float& operator[](int i) const {
        return this->values[i];
    }

    template<MemLoc M = Mem, std::enable_if_t<M != HOST, void*> = nullptr>
    float norm() const {
        float res;
        //myNorm(size, values, &res);
        cublasErrchk(cublasSnrm2(cublas_handle, size, values, 1, &res));
        gpuErrchk(cudaDeviceSynchronize());
        return res;
    }

    template<MemLoc M>
    friend std::ostream& operator<<(std::ostream& os, const Vector<M>& v);

    template<MemLoc M>
    friend class ColMajMatrix;
    
    template<MemLoc M>
    friend class SparseMatrix;

    template<MemLoc M>
    friend class Vector;

    template<MemLoc M>
    friend class SparseILU;

    template<MemLoc M>
    friend SparseMatrix<M> sparseApproximateMap(SparseMatrix<M> Ak, ColMajMatrix<M> A0);
    
    template<MemLoc M>
    friend SparseMatrix<M> sparseApproximateMap(const SparseMatrix<M>& Ak, const SparseMatrix<HOST>& A0); 
    
    int rows() const {
        return size;
    }

    template<MemLoc M = Mem>
    Vector<M> copy() const {
        Vector<M> x{size};
        Memcpy<Mem, M>{}(x.values, this->values, sizeof(float) * size);
        return std::move(x);
    }

    void truncate(int size_) {
        size = size_;
    }

private:

    int size = 0;
    float* values = nullptr;
    bool free = true;
    cusparseDnVecDescr_t desc = 0x0;
};
