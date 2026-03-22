#include "cudaerr.h"

template <class T>
class CudaLinear {
    T *memory;
public:
    CudaLinear(size_t size) {
        CSCT(cudaMalloc(&memory, sizeof(T) * size));
    }

    void memcpy(void *dst, int n, cudaMemcpyKind flag) {
        CSCT(cudaMemcpy(dst, memory, sizeof(T) * n, flag));
    }

    void memcpyToHost(void *dst, int n) {
        memcpy(dst, n, cudaMemcpyDeviceToHost);
    }

    void memcpyToDev(void *dst, int n) {
        memcpy(dst, n, cudaMemcpyHostToDevice);
    }

    T *get() {
        return memory;
    }

    ~CudaLinear() {
        CSC(cudaFree(memory));
    }
};
