#include <cublas_v2.h>
#include <cassert>
#include <cusparse.h>
#include <cusolverSp.h>
#include <iostream>

#pragma once

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

// https://stackoverflow.com/questions/13041399/equivalent-of-cudageterrorstring-for-cublas
const char* cublasGetErrorString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

#define cublasErrchk(ans)                                                         \
  { cublasAssert((ans), __FILE__, __LINE__); }

inline void cublasAssert(cublasStatus_t code, const char *file, int line,
                      bool abort = true) {
  if (code != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUBLASassert: %s %s %d\n", cublasGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

#define cusparseErrchk(ans)                                                         \
  { cusparseAssert((ans), __FILE__, __LINE__); }

inline void cusparseAssert(cusparseStatus_t code, const char *file, int line,
                      bool abort = true) {
  if (code != CUSPARSE_STATUS_SUCCESS) {
    fprintf(stderr, "CUSPARSEassert: %s %s %d\n", cusparseGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

#define cusolverErrchk(ans)                                                         \
  { cusolverAssert((ans), __FILE__, __LINE__); }

inline void cusolverAssert(cusolverStatus_t code, const char *file, int line,
                      bool abort = true) {
  if (code != CUSOLVER_STATUS_SUCCESS) {
      if(code == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED) {
        fprintf(stderr, "CUSOLVERassert: CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED %d %s %d\n", code, file, line);
      } else {
        fprintf(stderr, "CUSOLVERassert: %d %s %d\n", code, file, line);
      }
    if (abort)
      exit(code);
  }
}


cublasHandle_t cublas_handle;
cusparseHandle_t cusparse_handle;
cusolverSpHandle_t cusolver_handle;

void setup() {
    cublasErrchk(cublasCreate(&cublas_handle));
    cusparseErrchk(cusparseCreate(&cusparse_handle));
    cusolverErrchk(cusolverSpCreate(&cusolver_handle));
}

void cleanup() {
    cublasErrchk(cublasDestroy(cublas_handle));
    cusparseErrchk(cusparseDestroy(cusparse_handle));
    cusolverErrchk(cusolverSpDestroy(cusolver_handle));
}

enum MemLoc {
    UM,
    HOST,
    DEVICE
};

template<MemLoc FromLoc, MemLoc ToLoc>
struct Memcpy {
    template<typename T>
    inline void operator()(T* to, const T* from, size_t size) {
        memcpy(to, from, size);
    }
};

template<>
struct Memcpy<DEVICE, DEVICE> {
    template<typename T>
    inline void operator()(T* to, const T* from, size_t size) {
        gpuErrchk(cudaMemcpy(to, from, size, cudaMemcpyDeviceToDevice));
    }
};

template<>
struct Memcpy<HOST, DEVICE> {
    template<typename T>
    inline void operator()(T* to, const T* from, size_t size) {
        gpuErrchk(cudaMemcpy(to, from, size, cudaMemcpyHostToDevice));
    }
};

template<>
struct Memcpy<UM, DEVICE> {
    template<typename T>
    inline void operator()(T* to, const T* from, size_t size) {
        gpuErrchk(cudaMemcpy(to, from, size, cudaMemcpyHostToDevice));
    }
};

template<>
struct Memcpy<DEVICE, HOST> {
    template<typename T>
    inline void operator()(T* to, const T* from, size_t size) {
        gpuErrchk(cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost));
    }
};

template<>
struct Memcpy<DEVICE, UM> {
    template<typename T>
    inline void operator()(T* to, const T* from, size_t size) {
        gpuErrchk(cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost));
    }
};

template<MemLoc Loc>
struct Memset {
    template<typename T>
    inline void operator()(T* x, char c, size_t size) {
        memset(x, c, size);
    }
};

template<>
struct Memset<DEVICE> {
    template<typename T>
    inline void operator()(T* x, char c, size_t size) {
        gpuErrchk(cudaMemset(x, c, size));
    }
};

template<MemLoc Loc>
struct Alloc {
    template<typename T>
    inline void operator()(T** x, size_t elements) {
        gpuErrchk(cudaMallocManaged(x,  elements * sizeof(T)));
    }
};

template<>
struct Alloc<DEVICE> {
    template<typename T>
    inline void operator()(T** x, size_t elements) {
        gpuErrchk(cudaMalloc(x,  elements * sizeof(T)));
    }
};

template<>
struct Alloc<HOST> {
    template<typename T>
    inline void operator()(T** x, size_t elements) {
        *x = new T[elements];
    }
};

template<MemLoc Loc>
struct Dealloc {
    template<typename T>
    inline void operator()(T* x) {
        gpuErrchk(cudaFree(x));
    }
};

template<>
struct Dealloc<HOST> {
    template<typename T>
    inline void operator()(T* x) {
        delete[] x;
    }
};

// class names

template<MemLoc Mem>
struct SparseMatrix;

template<MemLoc Mem>
struct ColMajMatrix;

template<MemLoc Mem>
struct SparseILU;
