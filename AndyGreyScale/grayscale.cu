#include <iostream>
#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void convertToGrayscaleCUDA(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int greyOffset = row * width + col;
        int rgbOffset = greyOffset * channels;
        unsigned char r = inputImage[rgbOffset];
        unsigned char g = inputImage[rgbOffset + 1];
        unsigned char b = inputImage[rgbOffset + 2];
        outputImage[greyOffset] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

int main() {
    std::string inputFileName, outputFileName;
    std::cout << "Enter the name of the input image file (e.g., input.jpg): ";
    std::cin >> inputFileName;

    std::cout << "Enter the name of the output image file (e.g., output.jpg): ";
    std::cin >> outputFileName;

    int width, height, channels;
    unsigned char *img = stbi_load(inputFileName.c_str(), &width, &height, &channels, 0);
    if (!img) {
        std::cout << "Error in loading the image" << std::endl;
        return 1;
    }

    size_t imgSize = width * height * channels;
    size_t greyImgSize = width * height;
    unsigned char *d_inputImage, *d_outputImage;
    cudaMalloc(&d_inputImage, imgSize);
    cudaMalloc(&d_outputImage, greyImgSize);

    cudaMemcpy(d_inputImage, img, imgSize, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    convertToGrayscaleCUDA<<<dimGrid, dimBlock>>>(d_inputImage, d_outputImage, width, height, channels);

    unsigned char* greyImg = (unsigned char*)malloc(greyImgSize);
    cudaMemcpy(greyImg, d_outputImage, greyImgSize, cudaMemcpyDeviceToHost);

    stbi_write_jpg(outputFileName.c_str(), width, height, 1, greyImg, 100);

    stbi_image_free(img);
    free(greyImg);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return 0;
}
