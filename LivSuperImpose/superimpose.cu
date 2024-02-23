#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

__global__ void overlayKernel(unsigned char* largeImage, unsigned char* smallImage, int largeWidth, int smallHeight, int startX, int startY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= startX && x < (startX + smallWidth) && y >= startY && y < (startY + smallHeight)) {
        int largeOffset = (y * largeWidth + x) * 3; // Assuming RGB format
        int smallOffset = ((y - startY) * smallWidth + (x - startX)) * 3; // Assuming RGB format

        for (int i = 0; i < 3; ++i) { // RGB channels
            largeImage[largeOffset + i] = smallImage[smallOffset + i];
        }
    }
}

void overlayImages(const std::vector<std::string>& params) {
    // Parameters extraction and validation
    if (params.size() < 7) {
        std::cerr << "Insufficient parameters for overlay operation." << std::endl;
        return;
    }

    std::string largeImagePath = params[1];
    std::string smallImagePath = params[2];
    std::string outputImagePath = params[3];
    int startX = std::stoi(params[4]);
    int startY = std::stoi(params[5]);

    // Load images (Implement the loadPPM function)
    unsigned char* largeImage;
    unsigned char* smallImage;
    int largeWidth, largeHeight, smallWidth, smallHeight;
    loadPPM(largeImagePath, &largeImage, &largeWidth, &largeHeight);
    loadPPM(smallImagePath, &smallImage, &smallWidth, &smallHeight);

    // CUDA memory allocation and copying
    unsigned char *d_largeImage, *d_smallImage;
    cudaMalloc((void**)&d_largeImage, largeWidth * largeHeight * 3);
    cudaMalloc((void**)&d_smallImage, smallWidth * smallHeight * 3);
    cudaMemcpy(d_largeImage, largeImage, largeWidth * largeHeight * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_smallImage, smallImage, smallWidth * smallHeight * 3, cudaMemcpyHostToDevice);

    // Kernel launch parameters
    dim3 blockDim(16, 16);
    dim3 gridDim((largeWidth + blockDim.x - 1) / blockDim.x, (largeHeight + blockDim.y - 1) / blockDim.y);

    // Kernel invocation
    overlayKernel<<<gridDim, blockDim>>>(d_largeImage, d_smallImage, largeWidth, largeHeight, smallWidth, smallHeight, startX, startY);
    cudaDeviceSynchronize();

    // Copy back the result and free CUDA memory
    cudaMemcpy(largeImage, d_largeImage, largeWidth * largeHeight * 3, cudaMemcpyDeviceToHost);
    cudaFree(d_largeImage);
    cudaFree(d_smallImage);

    // Save the result (Implement the savePPM function)
    savePPM(outputImagePath, largeImage, largeWidth, largeHeight);

    // Free host memory
    delete[] largeImage;
    delete[] smallImage;
}
