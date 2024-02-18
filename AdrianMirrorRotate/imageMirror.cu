#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

__global__ void mirrorKernel(unsigned char *imageData, int width, int height, bool horizontal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= width || idy >= height) return;

    int rowSize = width * 3;

    if (horizontal) {
        if (idx < width / 2) {
            for (int byte = 0; byte < 3; ++byte) {
                int currentPos = idy * rowSize + idx * 3 + byte;
                int oppositePos = idy * rowSize + (width - 1 - idx) * 3 + byte;
                unsigned char temp = imageData[currentPos];
                imageData[currentPos] = imageData[oppositePos];
                imageData[oppositePos] = temp;
            }
        }
    } else {
        if (idy < height / 2) {
            for (int byte = 0; byte < 3; ++byte) {
                int currentPos = idy * rowSize + idx * 3 + byte;
                int oppositePos = (height - 1 - idy) * rowSize + idx * 3 + byte;
                unsigned char temp = imageData[currentPos];
                imageData[currentPos] = imageData[oppositePos];
                imageData[oppositePos] = temp;
            }
        }
    }
}

void mirrorImageCUDA(std::vector<unsigned char>& imageData, int width, int height, bool horizontal) {
    unsigned char *d_imageData;
    size_t imageSize = width * height * 3;
    cudaMalloc(&d_imageData, imageSize);
    cudaMemcpy(d_imageData, imageData.data(), imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    mirrorKernel<<<gridSize, blockSize>>>(d_imageData, width, height, horizontal);

    cudaMemcpy(imageData.data(), d_imageData, imageSize, cudaMemcpyDeviceToHost);
    cudaFree(d_imageData);
}

bool getUserChoice(const std::string& prompt) {
    std::string response;
    std::cout << prompt;
    std::getline(std::cin, response);
    return response[0] == 'Y' || response[0] == 'y';
}

int main() {
    std::string inputFilename = "input.ppm";
    std::ifstream file(inputFilename, std::ios::binary);
    std::string outputFilename = "mirrored.ppm";

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << inputFilename << std::endl;
        return 1;
    }

    std::string line;
    std::getline(file, line); // Read the magic number
    if (line != "P6") {
        std::cerr << "Unsupported format or not a P6 PPM file." << std::endl;
        return 1;
    }

    int width, height, maxval;
    file >> width >> height >> maxval;
    file.ignore(); // Skip the single whitespace after the maxval

    std::vector<unsigned char> imageData(width * height * 3);
    file.read(reinterpret_cast<char*>(imageData.data()), imageData.size());
    file.close();

    bool horizontalMirror = getUserChoice("Horizontal Mirror (Y/N)? ");
    if (horizontalMirror) {
        mirrorImageCUDA(imageData, width, height, true);
    }

    bool verticalMirror = getUserChoice("Vertical Mirror (Y/N)? ");
    if (verticalMirror) {
        mirrorImageCUDA(imageData, width, height, false);
    }

    std::ofstream outFile(outputFilename, std::ios::binary);
    outFile << "P6\n" << width << " " << height << "\n" << maxval << "\n";
    outFile.write(reinterpret_cast<const char*>(imageData.data()), imageData.size());
    outFile.close();

    return 0;
}
