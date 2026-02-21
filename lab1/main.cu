#include <iostream>

__global__ void kernel(double *dev_vecs, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    double vec1_elem = dev_vecs[idx];
    double vec2_elem = dev_vecs[idx + n];

    if (vec1_elem < vec2_elem) {
        dev_vecs[idx] = vec1_elem;
    } else {
        dev_vecs[idx] = vec2_elem;
    }
}

int main() {

    int n;

    std::cin >> n;
    if (n >= (1 << 25)) {
        fprintf(stderr, "n > 2^25");
        return 0;
    }

    double *vec1, *vec2, *dev_vecs;

    vec1 = (double *) malloc (sizeof(double) * n);
    if (!vec1) {
        fprintf(stderr, "malloc err");
        return 0;
    }

    vec2 = (double *) malloc (sizeof(double) * n);
    if (!vec2) {
        fprintf(stderr, "malloc err");
        free(vec1);
        return 0;
    }


    for (int i = 0; i < n; ++i) {
        std::cin >> vec1[i];
    }

    for (int i = 0; i < n; ++i) {
        std::cin >> vec2[i];
    }

    cudaMalloc(&dev_vecs, sizeof(double) * 2 * n);

    cudaMemcpy(dev_vecs, vec1, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vecs + n, vec2, sizeof(double) * n, cudaMemcpyHostToDevice);

    kernel<<<32, 32>>>(dev_vecs, n);

    cudaMemcpy(vec1, dev_vecs, sizeof(double) * n, cudaMemcpyDeviceToHost);

    cudaFree(dev_vecs);

    for (int i = 0; i < n; ++i) {
        printf("%.10e ", vec1[i]);
    }

    std::cout << std::endl;

    free(vec1);
    free(vec2);

    return 0;
}
