#include "cudaerr.h"

class CudaTexture {
    cudaTextureDesc textureDesc{};
    cudaResourceDesc resourceDesc{};
    cudaTextureObject_t texture = 0;
public:
    CudaTexture(cudaArray *array,
                cudaTextureAddressMode xAddressMode = cudaAddressModeWrap,
                cudaTextureAddressMode yAddressMode = cudaAddressModeWrap,
                cudaTextureAddressMode zAddressMode = cudaAddressModeWrap,
                cudaTextureFilterMode filterMode = cudaFilterModePoint) {

        memset(&resourceDesc, 0, sizeof(resourceDesc));
        resourceDesc.resType = cudaResourceTypeArray;
        resourceDesc.res.array.array = array;

        memset(&textureDesc, 0, sizeof(textureDesc));
        textureDesc.filterMode = filterMode;
        textureDesc.addressMode[0] = xAddressMode;
        textureDesc.addressMode[1] = yAddressMode;
        textureDesc.addressMode[2] = zAddressMode;
        textureDesc.readMode = cudaReadModeElementType;
        textureDesc.normalizedCoords = false;

        CSCT(cudaCreateTextureObject(&texture, &resourceDesc, &textureDesc, nullptr));
    }

    cudaTextureObject_t& get() {
        return texture;
    }

    ~CudaTexture() {
        CSC(cudaDestroyTextureObject(texture));
    }
};