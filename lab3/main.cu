#include "linear.cu"
#include "pixel.cu"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>


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

__constant__ Pixel<double> c_avgs[32];

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
        pixel.w = (unsigned char)class_id;
        out[x] = pixel;
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


        CudaLinear<uchar4> dev_image(image.size());

        dev_image.memcpyToDev(image.data(), image.size());

        CudaLinear<uchar4> dev_out(image.size());

        cudaMemcpyToSymbol(c_avgs, avgs.data(), sizeof(Pixel<double>) * num_classes);

        kernel<<<256, 256>>>(dev_image.get(), dev_out.get(), image.size(), num_classes);

        CSCT(cudaGetLastError());

        dev_out.memcpyToHost(image.data(), image.size());

        std::ofstream out(output_filename, std::ios::binary);
        if (out.bad() || !out.is_open())
            throw std::invalid_argument("Output file was not opened");

        out.write(reinterpret_cast<char*>(&w), sizeof(w));
        out.write(reinterpret_cast<char*>(&h), sizeof(h));
        if (!out)
            throw std::invalid_argument("Failed to write image dimensions");

        out.write(reinterpret_cast<char*>(image.data()), image.size() * sizeof(uchar4));
        if (!out)
            throw std::invalid_argument("Failed to write pixel data");
    } catch(const std::exception &e) {
        std::cerr << e.what() << std::endl;

        return 1;
    }

    return 0;
}