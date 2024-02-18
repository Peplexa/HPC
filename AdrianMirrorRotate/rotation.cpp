#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <cmath>
#include <atomic>
#include <string>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::mutex outputMutex;
std::atomic<int> completedRows(0);
int totalRows;

struct Pixel {
    unsigned char r, g, b;
};

double toRadians(double degrees) {
    return degrees * (M_PI / 180.0);
}

Pixel getPixel(const std::vector<unsigned char>& imageData, int width, int x, int y) {
    int index = (y * width + x) * 3;
    return {imageData[index], imageData[index + 1], imageData[index + 2]};
}

void setPixel(std::vector<unsigned char>& imageData, int width, int x, int y, const Pixel& pixel) {
    int index = (y * width + x) * 3;
    imageData[index] = pixel.r;
    imageData[index + 1] = pixel.g;
    imageData[index + 2] = pixel.b;
}

void rotateSection(const std::vector<unsigned char>& sourceImage, std::vector<unsigned char>& destImage,
                   int width, int height, double angle, int startRow, int endRow) {
    double rad = toRadians(angle);
    double sinAngle = std::sin(rad);
    double cosAngle = std::cos(rad);

    // Calculate the center of the image
    double centerX = width / 2.0;
    double centerY = height / 2.0;

    for (int y = startRow; y < endRow; ++y) {
        for (int x = 0; x < width; ++x) {
            // Translate point to origin
            double translatedX = x - centerX;
            double translatedY = y - centerY;

            // Rotate point
            double rotatedX = translatedX * cosAngle - translatedY * sinAngle;
            double rotatedY = translatedX * sinAngle + translatedY * cosAngle;

            // Translate point back
            int finalX = static_cast<int>(rotatedX + centerX);
            int finalY = static_cast<int>(rotatedY + centerY);

            if (finalX >= 0 && finalX < width && finalY >= 0 && finalY < height) {
                Pixel pixel = getPixel(sourceImage, width, x, y);
                setPixel(destImage, width, finalX, finalY, pixel);
            }
        }
        completedRows.fetch_add(1);
        std::lock_guard<std::mutex> lock(outputMutex);
        int progress = static_cast<int>((completedRows.load() / static_cast<double>(totalRows)) * 100);
        std::cout << "\rRotation Progress: " << progress << "%" << std::flush;
    }
}

void rotateImage(const std::vector<unsigned char>& sourceImage, std::vector<unsigned char>& destImage,
                 int originalWidth, int originalHeight, double angle, int numThreads) {
    int outputWidth = originalWidth * 2;
    int outputHeight = originalHeight * 2;
    destImage.resize(outputWidth * outputHeight * 3); // Resize and fill with black (0)

    double rad = toRadians(angle);
    double sinAngle = std::sin(rad);
    double cosAngle = std::cos(rad);

    // New center points for the larger image
    double newCenterX = outputWidth / 2.0;
    double newCenterY = outputHeight / 2.0;
    double originalCenterX = originalWidth / 2.0;
    double originalCenterY = originalHeight / 2.0;

    totalRows = originalHeight;

    std::vector<std::thread> threads;
    int rowsPerThread = originalHeight / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int startRow = i * rowsPerThread;
        int endRow = (i + 1) * rowsPerThread;
        if (i == numThreads - 1) {
            endRow = originalHeight;
        }
        threads.emplace_back([&](int startRow, int endRow) {
            for (int y = startRow; y < endRow; ++y) {
                for (int x = 0; x < originalWidth; ++x) {
                    // Adjust calculations for the center of the original image
                    double translatedX = x - originalCenterX;
                    double translatedY = y - originalCenterY;

                    double rotatedX = translatedX * cosAngle - translatedY * sinAngle;
                    double rotatedY = translatedX * sinAngle + translatedY * cosAngle;

                    // Adjust points to the center of the new, larger image
                    int finalX = static_cast<int>(rotatedX + newCenterX);
                    int finalY = static_cast<int>(rotatedY + newCenterY);

                    if (finalX >= 0 && finalX < outputWidth && finalY >= 0 && finalY < outputHeight) {
                        Pixel pixel = getPixel(sourceImage, originalWidth, x, y);
                        setPixel(destImage, outputWidth, finalX, finalY, pixel);
                    }
                }
            }
            completedRows.fetch_add(endRow - startRow);
            std::lock_guard<std::mutex> lock(outputMutex);
            int progress = static_cast<int>((completedRows.load() / static_cast<double>(totalRows)) * 100);
            std::cout << "\rRotation Progress: " << progress << "%" << std::flush;
        }, startRow, endRow);
    }

    for (auto& thread : threads) {
        thread.join();
    }
    std::cout << "\nFinished processing." << std::endl;
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

    int outputWidth = width * 2;
    int outputHeight = height * 2;
    std::vector<unsigned char> destImage(outputWidth * outputHeight * 3, 0); // Initialize with black background

    rotateImage(sourceImage, destImage, width, height, angle, 4);

    std::string outputFilename = "rotated.ppm";
    std::ofstream outputFile(outputFilename, std::ios::binary);
    if (!outputFile.is_open()) {
        std::cerr << "Error creating output file: " << outputFilename << std::endl;
        return 1;
    }

    outputFile << "P6\n" << outputWidth << " " << outputHeight << "\n" << maxval << "\n";
    outputFile.write(reinterpret_cast<const char*>(destImage.data()), destImage.size());
    outputFile.close();

    std::cout << "Rotation complete. Output saved to " << outputFilename << std::endl;

    return 0;
}