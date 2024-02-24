#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

struct Pixel {
    unsigned char r, g, b;
};

__device__ Pixel getPixelCUDA(const unsigned char* imageData, int width, int x, int y) {
    int index = (y * width + x) * 3;
    return {imageData[index], imageData[index + 1], imageData[index + 2]};
}

__device__ void setPixelCUDA(unsigned char* imageData, int width, int x, int y, const Pixel& pixel) {
    int index = (y * width + x) * 3;
    imageData[index] = pixel.r;
    imageData[index + 1] = pixel.g;
    imageData[index + 2] = pixel.b;
}

__global__ void overlayKernel(unsigned char* baseImage, unsigned char* overlayImage, int baseWidth, int overlayWidth, int overlayHeight, int posX, int posY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= posX && x < (posX + overlayWidth) && y >= posY && y < (posY + overlayHeight)) {
        Pixel overlayPixel = getPixelCUDA(overlayImage, overlayWidth, x - posX, y - posY);
        setPixelCUDA(baseImage, baseWidth, x, y, overlayPixel);
    }
}

void applyOverlayCUDA(const std::vector<std::string>& params) {
    if (params.size() < 6) {
        std::cerr << "Insufficient parameters provided for overlay." << std::endl;
        return;
    }

    std::string baseImageFilename = params[1];
    std::string overlayImageFilename = params[2];
    std::string outputFilename = params[3];
    int posX = std::stoi(params[4]);
    int posY = std::stoi(params[5]);

    int baseWidth, baseHeight, overlayWidth, overlayHeight, maxval;
    std::vector<unsigned char> baseImage, overlayImage;

    // Load the base and overlay images (loadPPM function to be defined)
    loadPPM(baseImageFilename, baseImage, baseWidth, baseHeight, maxval);
    loadPPM(overlayImageFilename, overlayImage, overlayWidth, overlayHeight, maxval);

    unsigned char *d_baseImage, *d_overlayImage;
    size_t baseImageSize = baseWidth * baseHeight * 3;
    size_t overlayImageSize = overlayWidth * overlayHeight * 3;

    cudaMalloc(&d_baseImage, baseImageSize);
    cudaMalloc(&d_overlayImage, overlayImageSize);
    cudaMemcpy(d_baseImage, baseImage.data(), baseImageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_overlayImage, overlayImage.data(), overlayImageSize, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((baseWidth + blockDim.x - 1) / blockDim.x, (baseHeight + blockDim.y - 1) / blockDim.y);
    overlayKernel<<<gridDim, blockDim>>>(d_baseImage, d_overlayImage, baseWidth, overlayWidth, overlayHeight, posX, posY);

    cudaMemcpy(baseImage.data(), d_baseImage, baseImageSize, cudaMemcpyDeviceToHost);
    cudaFree(d_baseImage);
    cudaFree(d_overlayImage);

    // Save the output image (savePPM function to be defined)
    savePPM(outputFilename, baseImage, baseWidth, baseHeight, maxval);
}

int main(int argc, char* argv[]) {
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty()) {
        std::cerr << "No arguments provided." << std::endl;
        return 1;
    }

    if (args[0] == "overlay") {
        applyOverlayCUDA(args);
    } else {
        std::cerr << "Unsupported transformation: " << args[0] << std::endl;
    }

    return 0;
}
