#include <cmath>
#include <iostream>
#include <cmath>

enum solutions {
    INCORRECT = -1,
    IMAGINARY = 0,
    ONE = 1,
    TWO = 2,
    ANY = 3
};

solutions solve_quadratic(float a, float b, float c, float *x1, float *x2) {
    if (a == 0) {
        if (b == 0) {
            if (c == 0)
                return ANY;
            return INCORRECT;
        }

        *x1 = -c / b;

        return ONE;
    }

    float double_a = 2 * a;

    float D = b * b - 2 * double_a * c;

    if (D < 0)
        return IMAGINARY;

    if (D == 0) {
        *x1 = -b / double_a;

        return ONE;
    }

    float sqrtD = std::sqrt(D);

    *x1 = (-b + sqrtD) / double_a;
    *x2 = (-b - sqrtD) / double_a;

    return TWO;
}

int main() {
    float a, b, c;

    std::cin >> a >> b >> c;

    float x1, x2;

    solutions solution = solve_quadratic(a, b, c, &x1, &x2);
    switch (solution) {
        case ANY:
            std::cout << "any";
            break;
        case INCORRECT:
            std::cout << "incorrect";
            break;
        case IMAGINARY:
            std::cout << "imaginary";
            break;
        case ONE:
            printf("%.6f", x1);
            break;
        case TWO:
            printf("%.6f %.6f", x1, x2);
            break;
    }

    std::cout << std::endl;

    return 0;
}