#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

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

__device__ Pixel bilinearInterpolate(const unsigned char* imageData, int width, int height, double x, double y) {
    int x1 = max(0, min(width - 1, static_cast<int>(floor(x))));
    int y1 = max(0, min(height - 1, static_cast<int>(floor(y))));
    int x2 = max(0, min(width - 1, x1 + 1));
    int y2 = max(0, min(height - 1, y1 + 1));

    Pixel Q11 = getPixelCUDA(imageData, width, x1, y1);
    Pixel Q21 = getPixelCUDA(imageData, width, x2, y1);
    Pixel Q12 = getPixelCUDA(imageData, width, x1, y2);
    Pixel Q22 = getPixelCUDA(imageData, width, x2, y2);

    double x2_x = x2 - x;
    double x_x1 = x - x1;
    double y2_y = y2 - y;
    double y_y1 = y - y1;

    double r = (Q11.r * x2_x * y2_y + Q21.r * x_x1 * y2_y + Q12.r * x2_x * y_y1 + Q22.r * x_x1 * y_y1) / ((x2 - x1) * (y2 - y1));
    double g = (Q11.g * x2_x * y2_y + Q21.g * x_x1 * y2_y + Q12.g * x2_x * y_y1 + Q22.g * x_x1 * y_y1) / ((x2 - x1) * (y2 - y1));
    double b = (Q11.b * x2_x * y2_y + Q21.b * x_x1 * y2_y + Q12.b * x2_x * y_y1 + Q22.b * x_x1 * y_y1) / ((x2 - x1) * (y2 - y1));

    return {static_cast<unsigned char>(r), static_cast<unsigned char>(g), static_cast<unsigned char>(b)};
}

__device__ void setPixelCUDA(unsigned char* imageData, int width, int x, int y, const Pixel& pixel) {
    int index = (y * width + x) * 3;
    imageData[index] = pixel.r;
    imageData[index + 1] = pixel.g;
    imageData[index + 2] = pixel.b;
}

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

__global__ void rotateKernel(const unsigned char* sourceImage, unsigned char* destImage, int originalWidth, int originalHeight, double cosAngle, double sinAngle, int outputWidth, int outputHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= outputWidth || y >= outputHeight) return;

    double centerX = originalWidth / 2.0;
    double centerY = originalHeight / 2.0;
    double newCenterX = outputWidth / 2.0;
    double newCenterY = outputHeight / 2.0;

    double translatedX = x - newCenterX;
    double translatedY = y - newCenterY;

    double originalX = translatedX * cosAngle + translatedY * sinAngle + centerX;
    double originalY = -translatedX * sinAngle + translatedY * cosAngle + centerY;

    if (originalX >= 0 && originalX < originalWidth && originalY >= 0 && originalY < originalHeight) {
        Pixel pixel = bilinearInterpolate(sourceImage, originalWidth, originalHeight, originalX, originalY);
        setPixelCUDA(destImage, outputWidth, x, y, pixel);
    } else {
        // Optional: Set pixels outside the original image area to a background color, if desired
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

void calculateRotatedDimensions(int originalWidth, int originalHeight, double angle, int& outputWidth, int& outputHeight) {
    double rad = angle * (M_PI / 180.0);
    double sinAngle = sin(rad);
    double cosAngle = cos(rad);

    double corners[4][2];
    double originX = originalWidth / 2.0;
    double originY = originalHeight / 2.0;

    // Calculate rotated positions of the corners
    for (int i = 0; i < 4; ++i) {
        double x = (i % 2) * originalWidth - originX;
        double y = (i / 2) * originalHeight - originY;
        corners[i][0] = cosAngle * x - sinAngle * y + originX;
        corners[i][1] = sinAngle * x + cosAngle * y + originY;
    }

    double minX = corners[0][0], maxX = corners[0][0], minY = corners[0][1], maxY = corners[0][1];
    for (int i = 1; i < 4; ++i) {
        minX = min(minX, corners[i][0]);
        maxX = max(maxX, corners[i][0]);
        minY = min(minY, corners[i][1]);
        maxY = max(maxY, corners[i][1]);
    }

    outputWidth = static_cast<int>(ceil(maxX - minX));
    outputHeight = static_cast<int>(ceil(maxY - minY));
}

void rotateImageCUDA(const std::vector<unsigned char>& sourceImage, std::vector<unsigned char>& destImage, int originalWidth, int originalHeight, double angle, int& outputWidth, int& outputHeight) {
    calculateRotatedDimensions(originalWidth, originalHeight, angle, outputWidth, outputHeight);
    destImage.resize(outputWidth * outputHeight * 3);

    double rad = angle * (M_PI / 180.0);
    double sinAngle = sin(rad);
    double cosAngle = cos(rad);

    unsigned char *d_sourceImage, *d_destImage;
    size_t sourceSize = originalWidth * originalHeight * 3;
    size_t destSize = outputWidth * outputHeight * 3;
    cudaMalloc(&d_sourceImage, sourceSize);
    cudaMalloc(&d_destImage, destSize);
    cudaMemcpy(d_sourceImage, sourceImage.data(), sourceSize, cudaMemcpyHostToDevice);
    cudaMemset(d_destImage, 0, destSize);

    dim3 blockSize(16, 16);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x, (outputHeight + blockSize.y - 1) / blockSize.y);
    rotateKernel<<<gridSize, blockSize>>>(d_sourceImage, d_destImage, originalWidth, originalHeight, cosAngle, sinAngle, outputWidth, outputHeight);

    cudaMemcpy(destImage.data(), d_destImage, destSize, cudaMemcpyDeviceToHost);
    cudaFree(d_sourceImage);
    cudaFree(d_destImage);
}

void applyTransformation(const std::vector<std::string>& params) {
    if (params.size() < 3) {
        std::cerr << "Insufficient parameters provided." << std::endl;
        return;
    }

    std::string transformationType = params[0];
    std::string inputFilename = params[1];
    std::string outputFilename = params[2];

    // Read input image
    std::ifstream file(inputFilename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << inputFilename << std::endl;
        return;
    }

    std::string line;
    std::getline(file, line); // PPM magic number
    if (line != "P6") {
        std::cerr << "Unsupported format or not a P6 PPM file." << std::endl;
        file.close();
        return;
    }

    int width, height, maxval;
    file >> width >> height >> maxval;
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::vector<unsigned char> imageData(width * height * 3);
    file.read(reinterpret_cast<char*>(imageData.data()), imageData.size());
    file.close();

    if (transformationType == "mirror") {
        if (params.size() < 4) {
            std::cerr << "Mirror direction not specified." << std::endl;
            return;
        }
        bool horizontal = params[3] == "horizontal";
        mirrorImageCUDA(imageData, width, height, horizontal);
    } else if (transformationType == "rotate") {
        if (params.size() < 4) {
            std::cerr << "Rotation angle not specified." << std::endl;
            return;
        }
        double angle = std::stod(params[3]);
        int outputWidth, outputHeight;
        std::vector<unsigned char> destImage;
        rotateImageCUDA(imageData, destImage, width, height, angle, outputWidth, outputHeight);
        imageData.swap(destImage); // Use rotated image for output
        width = outputWidth;
        height = outputHeight;
    } else {
        std::cerr << "Unsupported transformation type." << std::endl;
        return;
    }

    // Write the processed image to the output file
    std::ofstream outFile(outputFilename, std::ios::binary);
    outFile << "P6\n" << width << " " << height << "\n" << maxval << "\n";
    outFile.write(reinterpret_cast<const char*>(imageData.data()), imageData.size());
    outFile.close();
}

int main() {
    // Example parameters for demonstration
    //std::vector<std::string> params = {"rotate", "input.ppm", "output.ppm", "90"};
    std::vector<std::string> params = {"mirror", "input.ppm", "output.ppm", "horizontal"};
    
    applyTransformation(params);

    return 0;
}
