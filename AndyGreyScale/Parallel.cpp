#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <string>
#include <fstream>
#include <chrono> // Include chrono for timing
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void convertToGrayscale(unsigned char* image, int width, int height, int channels, int start, int end) {
    int pixelCount = width * height;
    for (int i = start; i < end; ++i) {
        int idx = i * channels;
        unsigned char r = image[idx];
        unsigned char g = image[idx + 1];
        unsigned char b = image[idx + 2];
        unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
        image[idx] = image[idx + 1] = image[idx + 2] = gray;
    }
}

bool fileExists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

int main() {
    std::string inputFileName, outputFileName;
    inputFileName = "your_image.jpg";
    outputFileName = "grey.jpg";

    int width, height, channels;
    unsigned char* img = stbi_load(inputFileName.c_str(), &width, &height, &channels, 0);
    if (!img) {
        std::cout << "Error in loading the image" << std::endl;
        return 1;
    }

    int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int blockSize = (width * height) / numThreads;

    auto start_time = std::chrono::high_resolution_clock::now(); // Start timing

    for (int i = 0; i < numThreads; ++i) {
        int start = i * blockSize;
        int end = (i + 1) * blockSize;
        if (i == numThreads - 1) {
            end = width * height;
        }
        threads.emplace_back(convertToGrayscale, img, width, height, channels, start, end);
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now(); // End timing
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    stbi_write_jpg(outputFileName.c_str(), width, height, channels, img, 100);
    stbi_image_free(img);

    std::cout << "Conversion to grayscale completed successfully in " << duration.count() << " milliseconds." << std::endl;

    return 0;
}
