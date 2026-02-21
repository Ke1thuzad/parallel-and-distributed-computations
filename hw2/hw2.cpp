#include <iostream>

void bubble_sort(float *arr, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }
}

int main() {
    int n;

    std::cin >> n;

    float *arr = (float *) malloc(sizeof(float) * n);
    if (!arr) {
        fprintf(stderr, "malloc err");
        return 0;
    }

    float temp;
    for (int i = 0; i < n; ++i) {
        std::cin >> temp;
        arr[i] = temp;
    }

    bubble_sort(arr, n);

    for (int i = 0; i < n; ++i) {
        printf("%.6e ", arr[i]);
    }

    std::cout << std::endl;

    free(arr);

    return 0;
}