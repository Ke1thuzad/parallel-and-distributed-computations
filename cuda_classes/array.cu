#include "cudaerr.h"

template <class T>
class CudaArray {
    cudaArray *array = nullptr;
    int width, height;
public:
    explicit CudaArray(size_t width, size_t height = 0) : width(width), height(height) {
        cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<T>();
        CSCT(cudaMallocArray(&array, &channelFormatDesc, width, height));
    }

    cudaArray* get() {
        return array;
    }

    void memcpy2D(const void *data, cudaMemcpyKind flag, size_t wOffset = 0, size_t hOffset = 0) {
        CSCT(cudaMemcpy2DToArray(array, wOffset, hOffset, data, width * sizeof(T), width * sizeof(T), height, flag));
    }

    void memcpy2D_toDevice(const void *data, size_t wOffset = 0, size_t hOffset = 0) {
        memcpy2D(data, cudaMemcpyHostToDevice, wOffset, hOffset);
    }

    void memcpy2D_toHost(const void *data, size_t wOffset = 0, size_t hOffset = 0) {
        memcpy2D(data, cudaMemcpyDeviceToHost, wOffset, hOffset);
    }

    ~CudaArray() {
        CSC(cudaFreeArray(array));
    }
};