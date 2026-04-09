#ifndef PARALLEL_AND_DISTRIBUTED_COMPUTATIONS_PIXEL_CU
#define PARALLEL_AND_DISTRIBUTED_COMPUTATIONS_PIXEL_CU

#include <cmath>
#include <iostream>

template<class T>
struct Pixel {
    T r, g, b, a;

    Pixel() = default;

    __host__ __device__
    Pixel(T r, T g, T b, T a) : r(r), g(g), b(b), a(a) { }

    __host__ __device__
    explicit Pixel(uchar4 p) : r(p.x), g(p.y), b(p.z), a(p.w) { }

    __host__ __device__
    T length() const {
        return std::sqrt(r * r + g * g + b * b  );
    }

    __host__ __device__
    Pixel operator+(const Pixel &other) {
        return Pixel{r + other.r, g + other.g, b + other.b, a + other.a};
    }

    __host__ __device__
    Pixel &operator+=(const Pixel &other) {
        *this = *this + other;

        return *this;
    }

    __host__ __device__
    Pixel operator*(const Pixel &other) const {
        return Pixel{r * other.r, g * other.g, b * other.b, a * other.a};
    }

    __host__ __device__
    Pixel operator/(const Pixel &other) const {
        return Pixel{r / other.r, g / other.g, b / other.b, a / other.a};
    }

    __host__ __device__
    Pixel operator*(T other) const {
        return Pixel{r * other, g * other, b * other, a * other};
    }

    __host__ __device__
    Pixel operator/(T other) const {
        return Pixel{r / other, g / other, b / other, a / other};
    }

    __host__ __device__
    T dot(const Pixel &other) const {
        return r * other.r + g * other.g + b * other.b;
    }

    __host__ __device__
    static Pixel normalized(const Pixel &pixel) {
        T l = pixel.length();
        if (l < 1e-9f)
            return Pixel(0, 0, 0, 0);

        return pixel / l;
    }

    __host__ __device__
    Pixel& normalize() {
        *this = normalized(*this);
        return *this;
    }

    friend std::ostream &operator <<(std::ostream &stream, Pixel &pixel) {
        stream << "Pixel(" << pixel.r << "," << pixel.g << "," << pixel.b << "," << pixel.a << ")";

        return stream;
    }
};

#endif //PARALLEL_AND_DISTRIBUTED_COMPUTATIONS_PIXEL_CU
