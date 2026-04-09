#include "linear.cu"
#include "pixel.cu"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>

__constant__ uchar4 c_palette[32];
__constant__ Pixel<double> c_avgs[32];

__host__ std::vector<uchar4> read_image(const std::string &input_filename, int &w, int &h) {
    std::ifstream img_file(input_filename, std::ios::binary);
    if (img_file.bad() || !img_file.is_open())
        throw std::invalid_argument("Input file was not opened");

    img_file.read(reinterpret_cast<char*>(&w), sizeof(w));
    img_file.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!img_file)
        throw std::invalid_argument("Failed to read image dimensions");

    std::vector<uchar4> data(w * h);

    img_file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(uchar4));
    if (!img_file)
        throw std::invalid_argument("Failed to read pixel data");

    return data;
}

__host__ void get_averages(int num_classes, std::vector<uchar4> &image, int w, std::vector<Pixel<double>> &avgs) {
    for (int k = 0; k < num_classes; ++k) {
        int class_n = 0;
        Pixel<double> sum{};

        std::cin >> class_n;

        for (int i = 0; i < class_n; ++i) {
            int x, y;
            std::cin >> x >> y;
            sum += Pixel<double>(image[x + y * w]);
        }
        avgs.push_back(Pixel<double>::normalized(sum / class_n));
    }
}

__device__ int classify(uchar4 pixel, int num_classes) {
    int class_res = 0;
    double max_val = -1e20f;

    Pixel<double> pix(pixel);

    for (int k = 0; k < num_classes; ++k) {
        double cur_dot = pix.dot(c_avgs[k]);

        if (cur_dot > max_val) {
            max_val = cur_dot;
            class_res = k;
        }
    }
    return class_res;
}

__global__ void kernel(uchar4 *image, uchar4 *out, size_t wh, int num_classes) {
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t offsetX = blockDim.x * gridDim.x;

    for (uint32_t x = idx; x < wh; x += offsetX) {
        uchar4 pixel = image[x];
        int class_id = classify(pixel, num_classes);
        out[x] = c_palette[class_id];
    }
}

int main() {
    try {
        int w, h;
        std::string input_filename, output_filename;
        std::cin >> input_filename >> output_filename;

        std::vector<uchar4> image = read_image(input_filename, w, h);

        int num_classes = 0;
        std::cin >> num_classes;
        if (num_classes < 1 || num_classes > 32)
            throw std::invalid_argument("Invalid class amount");

        std::vector<Pixel<double>> avgs;
        get_averages(num_classes, image, w, avgs);

        std::vector<uchar4> h_palette(32);
        h_palette[0] = {255, 0, 0, 255};
        h_palette[1] = {0, 255, 0, 255};
        h_palette[2] = {0, 0, 255, 255};
        h_palette[3] = {255, 255, 0, 255};
        h_palette[4] = {255, 0, 255, 255};
        h_palette[5] = {0, 255, 255, 255};
        h_palette[6] = {255, 128, 0, 255};
        h_palette[7] = {128, 0, 255, 255};
        h_palette[8] = {128, 128, 128, 255};
        h_palette[9] = {0, 128, 0, 255};
        for(int i = 10; i < 32; ++i) {
            h_palette[i] = {(unsigned char)(i * 7), (unsigned char)(i * 13), (unsigned char)(i * 21), 255};
        }

        CudaLinear<uchar4> dev_image(image.size());
        dev_image.memcpyToDev(image.data(), image.size());

        CudaLinear<uchar4> dev_out(image.size());

        cudaMemcpyToSymbol(c_avgs, avgs.data(), sizeof(Pixel<double>) * num_classes);
        cudaMemcpyToSymbol(c_palette, h_palette.data(), sizeof(uchar4) * 32);

        kernel<<<256, 256>>>(dev_image.get(), dev_out.get(), image.size(), num_classes);

        CSCT(cudaGetLastError());

        dev_out.memcpyToHost(image.data(), image.size());

        std::ofstream out(output_filename, std::ios::binary);
        if (out.bad() || !out.is_open())
            throw std::invalid_argument("Output file was not opened");

        out.write(reinterpret_cast<char*>(&w), sizeof(w));
        out.write(reinterpret_cast<char*>(&h), sizeof(h));
        out.write(reinterpret_cast<char*>(image.data()), image.size() * sizeof(uchar4));

    } catch(const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}