#include "array.cu"
#include "linear.cu"
#include "texture.cu"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

__device__ float rgb_to_y(uchar4 rgb) {
    return 0.299f * rgb.x + 0.587f * rgb.y + 0.114 * rgb.z;
}

__device__ int clamp(int x, int mn, int mx) {
    if (x < mn) {
        return mn;
    } else if (x > mx) {
        return mx;
    }
    return x;
}

__global__ void kernel(cudaTextureObject_t texture, uchar4 *out, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;
    int x, y;

    char kernelX[3][3] = {{-1, 0, 1},
                          {-2, 0, 2},
                          {-1, 0, 1}};

    uchar4 pixel;

    for (y = idy; y < h; y += offsetY) {
        for (x = idx; x < w; x += offsetX) {
            float sumX = 0, sumY = 0;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    pixel = tex2D<uchar4>(texture, x + i - 0.5f, y + j - 0.5f);
                    float yp = rgb_to_y(pixel);
                    sumX += yp * kernelX[j][i];
                    sumY += yp * kernelX[i][j];
                }
            }
            pixel = tex2D<uchar4>(texture, x + 0.5f, y + 0.5f);

            float rootedSum = sqrtf(sumX * sumX + sumY * sumY);
            unsigned char res = clamp(rootedSum, 0, 255);

            out[(size_t)y * w + x] = make_uchar4(res, res, res, pixel.w);
        }
    }
}

int main() {
    try {
        int w, h;

        std::string input_filename, output_filename;
        std::cin >> input_filename >> output_filename;

        std::ifstream img_file(input_filename, std::ios::binary);
        if (img_file.bad() || !img_file.is_open())
            throw std::invalid_argument("Input file was not opened");

        img_file.read(reinterpret_cast<char*>(&w), sizeof(w));
        img_file.read(reinterpret_cast<char*>(&h), sizeof(h));
        if (!img_file)
            throw std::invalid_argument("Failed to read image dimensions");

        size_t total_pixels = (size_t)w * h;

        std::vector<uchar4> data(total_pixels);
        img_file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(uchar4));
        if (!img_file)
            throw std::invalid_argument("Failed to read pixel data");

        CudaArray<uchar4> array(w, h);

        array.memcpy2D_toDevice(data.data());

        CudaTexture cudaTexture(array.get(), cudaAddressModeClamp, cudaAddressModeClamp);

        CudaLinear<uchar4> dev_out(total_pixels);

        kernel<<<dim3(16, 16), dim3(32, 32)>>>(cudaTexture.get(), dev_out.get(), w, h);

        CSCT(cudaGetLastError());

        dev_out.memcpyToHost(data.data(), total_pixels);

        std::ofstream out(output_filename, std::ios::binary);
        if (out.bad() || !out.is_open())
            throw std::invalid_argument("Output file was not opened");

        out.write(reinterpret_cast<char*>(&w), sizeof(w));
        out.write(reinterpret_cast<char*>(&h), sizeof(h));
        if (!out)
            throw std::invalid_argument("Failed to write image dimensions");

        out.write(reinterpret_cast<char*>(data.data()), data.size() * sizeof(uchar4));
        if (!out)
            throw std::invalid_argument("Failed to write pixel data");
    } catch(const std::exception &e) {
        std::cerr << e.what() << std::endl;

        return 1;
    }

    return 0;
}