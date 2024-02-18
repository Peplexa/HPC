#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

__global__ void rotateKernel(const unsigned char* sourceImage, unsigned char* destImage,
                             int originalWidth, int originalHeight, double cosAngle, double sinAngle,
                             int outputWidth, int outputHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= outputWidth || y >= outputHeight) return;

    double centerX = originalWidth / 2.0;
    double centerY = originalHeight / 2.0;
    double newCenterX = outputWidth / 2.0;
    double newCenterY = outputHeight / 2.0;

    // Translate back to original image coordinates
    double translatedX = x - newCenterX;
    double translatedY = y - newCenterY;

    // Apply rotation
    double originalX = translatedX * cosAngle + translatedY * sinAngle + centerX;
    double originalY = -translatedX * sinAngle + translatedY * cosAngle + centerY;

    if (originalX >= 0 && originalX < originalWidth && originalY >= 0 && originalY < originalHeight) {
        Pixel pixel = getPixelCUDA(sourceImage, originalWidth, int(originalX), int(originalY));
        setPixelCUDA(destImage, outputWidth, x, y, pixel);
    }
}

void rotateImageCUDA(const std::vector<unsigned char>& sourceImage, std::vector<unsigned char>& destImage,
                     int originalWidth, int originalHeight, double angle) {
    int outputWidth = originalWidth * 2;
    int outputHeight = originalHeight * 2;
    destImage.resize(outputWidth * outputHeight * 3); // Resize and fill with black (0)

    double rad = angle * (M_PI / 180.0);
    double sinAngle = std::sin(rad);
    double cosAngle = std::cos(rad);

    unsigned char *d_sourceImage, *d_destImage;
    size_t sourceSize = originalWidth * originalHeight * 3;
    size_t destSize = outputWidth * outputHeight * 3;
    cudaMalloc(&d_sourceImage, sourceSize);
    cudaMalloc(&d_destImage, destSize);
    cudaMemcpy(d_sourceImage, sourceImage.data(), sourceSize, cudaMemcpyHostToDevice);
    cudaMemset(d_destImage, 0, destSize); // Initialize destination image to black

    dim3 blockSize(16, 16);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x, (outputHeight + blockSize.y - 1) / blockSize.y);
    rotateKernel<<<gridSize, blockSize>>>(d_sourceImage, d_destImage, originalWidth, originalHeight, cosAngle, sinAngle, outputWidth, outputHeight);

    cudaMemcpy(destImage.data(), d_destImage, destSize, cudaMemcpyDeviceToHost);
    cudaFree(d_sourceImage);
    cudaFree(d_destImage);
}

int main() {
    std::string inputFilename = "input.ppm";
    std::ifstream inputFile(inputFilename, std::ios::binary);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file: " << inputFilename << std::endl;
        return 1;
    }

    std::string header;
    int width, height, maxval;
    inputFile >> header;
    if (header != "P6") {
        std::cerr << "Unsupported format or not a P6 PPM file." << std::endl;
        inputFile.close();
        return 1;
    }
    inputFile >> width >> height >> maxval;
    inputFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Skip to the end of the header line

    std::vector<unsigned char> sourceImage(width * height * 3);
    inputFile.read(reinterpret_cast<char*>(sourceImage.data()), sourceImage.size());
    inputFile.close();

    double angle;
    std::cout << "Enter the rotation angle in degrees: ";
    std::cin >> angle;

    std::vector<unsigned char> destImage; // Destination image will be resized inside rotateImageCUDA
    rotateImageCUDA(sourceImage, destImage, width, height, angle);

    std::string outputFilename = "rotated.ppm";
    std::ofstream outputFile(outputFilename, std::ios::binary);
    if (!outputFile.is_open()) {
        std::cerr << "Error creating output file: " << outputFilename << std::endl;
        return 1;
    }

    int outputWidth = width * 2;
    int outputHeight = height * 2;
    outputFile << "P6\n" << outputWidth << " " << outputHeight << "\n" << maxval << "\n";
    outputFile.write(reinterpret_cast<const char*>(destImage.data()), destImage.size());
    outputFile.close();

    std::cout << "Rotation complete. Output saved to " << outputFilename << std::endl;

    return 0;
}
