#include "cudaerr.h"

template <class T>
class CudaLinear {
    T *memory = nullptr;
    size_t size = 0;
public:
    CudaLinear(size_t size) : size(size) {
        CSCT(cudaMalloc(&memory, sizeof(T) * size));
    }

    CudaLinear(const CudaLinear&) = delete;
    CudaLinear& operator=(const CudaLinear&) = delete;

    void memcpy(void *dst, void *src, size_t count, cudaMemcpyKind flag) {
        CSCT(cudaMemcpy(dst, src, sizeof(T) * count, flag));
    }

    void memcpyToHost(void *dst, size_t count) {
        memcpy(dst, memory, count, cudaMemcpyDeviceToHost);
    }

    void memcpyToDev(void *src, size_t count) {
        memcpy(memory, src, count, cudaMemcpyHostToDevice);
    }

    T *get() {
        return memory;
    }

    ~CudaLinear() {
        CSC(cudaFree(memory));
    }
};
