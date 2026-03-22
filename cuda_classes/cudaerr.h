#ifndef PARALLEL_AND_DISTRIBUTED_COMPUTATIONS_CUDAERR_H
#define PARALLEL_AND_DISTRIBUTED_COMPUTATIONS_CUDAERR_H

#include <stdexcept>
#include <iostream>
#include <string>

#define CSCT(call)                                                      \
    do {                                                                \
        cudaError_t res = call;                                         \
        if (res != cudaSuccess) {                                       \
            std::string msg = "ERROR in " + std::string(__FILE__) + ":" \
                            + std::to_string(__LINE__) + ". Message: "  \
                            + cudaGetErrorString(res) + "\n";           \
            throw std::runtime_error(msg);                              \
        }                                                               \
    } while(0)

#define CSC(call)                                                       \
    do {                                                                \
        cudaError_t res = call;                                         \
        if (res != cudaSuccess) {                                       \
            std::string msg = "ERROR in " + std::string(__FILE__) + ":" \
                            + std::to_string(__LINE__) + ". Message: "  \
                            + cudaGetErrorString(res) + "\n";           \
            std::cerr << msg << std::endl;                              \
        }                                                               \
    } while(0)

#endif //PARALLEL_AND_DISTRIBUTED_COMPUTATIONS_CUDAERR_H
