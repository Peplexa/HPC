#include <cuda_runtime.h>
#include "superimpose.h"

__global__ void overlayKernel(unsigned char* largeImage, unsigned char* smallImage, int largeWidth, int largeHeight, int smallWidth, int smallHeight, int startX, int startY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= startX && x < (startX + smallWidth) && y >= startY && y < (startY + smallHeight)) {
        int largeOffset = (y * largeWidth + x) * 4;
        int smallOffset = ((y - startY) * smallWidth + (x - startX)) * 4;
        
        for (int i = 0; i < 4; ++i) {
            largeImage[largeOffset + i] = smallImage[smallOffset + i];
        }
    }
}

void overlayImages(unsigned char* largeImage, unsigned char* smallImage, int largeWidth, int largeHeight, int smallWidth, int smallHeight, int startX, int startY) {
    unsigned char *d_largeImage, *d_smallImage;
    cudaMalloc((void**)&d_largeImage, largeWidth * largeHeight * 4);
    cudaMalloc((void**)&d_smallImage, smallWidth * smallHeight * 4);

    cudaMemcpy(d_largeImage, largeImage, largeWidth * largeHeight * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_smallImage, smallImage, smallWidth * smallHeight * 4, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((largeWidth + blockDim.x - 1) / blockDim.x, (largeHeight + blockDim.y - 1) / blockDim.y);

    overlayKernel<<<gridDim, blockDim>>>(d_largeImage, d_smallImage, largeWidth, largeHeight, smallWidth, smallHeight, startX, startY);
    cudaDeviceSynchronize();

    cudaMemcpy(largeImage, d_largeImage, largeWidth * largeHeight * 4, cudaMemcpyDeviceToHost);

    cudaFree(d_largeImage);
    cudaFree(d_smallImage);
}
