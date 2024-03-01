#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <limits>
#include <filesystem>
namespace fs = std::filesystem;


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Pixel {
    unsigned char r, g, b;
};

std::mutex outputMutex;
std::atomic<int> completedRows(0);
int totalRows;

Pixel getPixel(const std::vector<unsigned char>& imageData, int width, int x, int y) {
    int index = (y * width + x) * 3;
    return {imageData[index], imageData[index + 1], imageData[index + 2]};
}

Pixel bilinearInterpolate(const std::vector<unsigned char>& imageData, int width, int height, double x, double y) {
    int x1 = std::max(0, std::min(width - 1, static_cast<int>(std::floor(x))));
    int y1 = std::max(0, std::min(height - 1, static_cast<int>(std::floor(y))));
    int x2 = std::max(0, std::min(width - 1, x1 + 1));
    int y2 = std::max(0, std::min(height - 1, y1 + 1));

    Pixel Q11 = getPixel(imageData, width, x1, y1);
    Pixel Q21 = getPixel(imageData, width, x2, y1);
    Pixel Q12 = getPixel(imageData, width, x1, y2);
    Pixel Q22 = getPixel(imageData, width, x2, y2);

    double x2_x = x2 - x;
    double x_x1 = x - x1;
    double y2_y = y2 - y;
    double y_y1 = y - y1;

    double r = (Q11.r * x2_x * y2_y + Q21.r * x_x1 * y2_y + Q12.r * x2_x * y_y1 + Q22.r * x_x1 * y_y1) / ((x2 - x1) * (y2 - y1));
    double g = (Q11.g * x2_x * y2_y + Q21.g * x_x1 * y2_y + Q12.g * x2_x * y_y1 + Q22.g * x_x1 * y_y1) / ((x2 - x1) * (y2 - y1));
    double b = (Q11.b * x2_x * y2_y + Q21.b * x_x1 * y2_y + Q12.b * x2_x * y_y1 + Q22.b * x_x1 * y_y1) / ((x2 - x1) * (y2 - y1));

    return {static_cast<unsigned char>(r), static_cast<unsigned char>(g), static_cast<unsigned char>(b)};
}

void updateProgress(const std::string& operation) {
    std::lock_guard<std::mutex> lock(outputMutex);
    int progress = static_cast<int>((completedRows.load() / static_cast<double>(totalRows)) * 100);
    std::cout << "\r" << operation << " Progress: " << progress << "%" << std::flush;
}

void mirrorSection(std::vector<unsigned char>& imageData, int width, int height, int startRow, int endRow, bool horizontal) {
    int rowSize = width * 3; // 3 bytes per pixel (RGB)
    for (int y = startRow; y < endRow; ++y) {
        for (int x = 0; x < (horizontal ? width / 2 : width); ++x) {
            int oppositeX = width - 1 - x;
            int oppositeY = height - 1 - y;

            if (horizontal) {
                for (int byte = 0; byte < 3; ++byte) { // Swap RGB bytes for horizontal mirror
                    std::swap(imageData[y * rowSize + x * 3 + byte], imageData[y * rowSize + oppositeX * 3 + byte]);
                }
            } else if (y < height / 2) { // For vertical mirror, only process the top half of the image
                for (int byte = 0; byte < 3; ++byte) {
                    std::swap(imageData[y * rowSize + x * 3 + byte], imageData[oppositeY * rowSize + x * 3 + byte]);
                }
            }
        }
        completedRows.fetch_add(1);
        updateProgress(horizontal ? "Horizontal Mirror" : "Vertical Mirror");
    }
}

void calculateRotatedDimensions(int originalWidth, int originalHeight, double angle, int& outputWidth, int& outputHeight) {
    double rad = angle * (M_PI / 180.0);
    double sinAngle = std::sin(rad);
    double cosAngle = std::cos(rad);

    double halfWidth = originalWidth / 2.0;
    double halfHeight = originalHeight / 2.0;

    double cornersX[4], cornersY[4];
    cornersX[0] = -halfWidth * cosAngle - -halfHeight * sinAngle;
    cornersY[0] = -halfWidth * sinAngle + -halfHeight * cosAngle;
    cornersX[1] = halfWidth * cosAngle - -halfHeight * sinAngle;
    cornersY[1] = halfWidth * sinAngle + -halfHeight * cosAngle;
    cornersX[2] = -halfWidth * cosAngle - halfHeight * sinAngle;
    cornersY[2] = -halfWidth * sinAngle + halfHeight * cosAngle;
    cornersX[3] = halfWidth * cosAngle - halfHeight * sinAngle;
    cornersY[3] = halfWidth * sinAngle + halfHeight * cosAngle;

    double minX = *std::min_element(cornersX, cornersX + 4);
    double maxX = *std::max_element(cornersX, cornersX + 4);
    double minY = *std::min_element(cornersY, cornersY + 4);
    double maxY = *std::max_element(cornersY, cornersY + 4);

    outputWidth = static_cast<int>(std::ceil(maxX - minX));
    outputHeight = static_cast<int>(std::ceil(maxY - minY));
}

void rotateFunction(const std::vector<unsigned char>& sourceImage, std::vector<unsigned char>& destImage, int originalWidth, int originalHeight, double cosAngle, double sinAngle, int outputWidth, int outputHeight, int startY, int endY) {
    double centerX = originalWidth / 2.0;
    double centerY = originalHeight / 2.0;
    double newCenterX = outputWidth / 2.0;
    double newCenterY = outputHeight / 2.0;

    for (int y = startY; y < endY; ++y) {
        for (int x = 0; x < outputWidth; ++x) {
            double translatedX = x - newCenterX;
            double translatedY = y - newCenterY;

            double originalX = translatedX * cosAngle + translatedY * sinAngle + centerX;
            double originalY = -translatedX * sinAngle + translatedY * cosAngle + centerY;

            if (originalX >= 0 && originalX < originalWidth && originalY >= 0 && originalY < originalHeight) {
                Pixel pixel = bilinearInterpolate(sourceImage, originalWidth, originalHeight, originalX, originalY);
                int index = (y * outputWidth + x) * 3;
                destImage[index] = pixel.r;
                destImage[index + 1] = pixel.g;
                destImage[index + 2] = pixel.b;
            }
        }
    }
}

void mirrorImage(std::vector<unsigned char>& imageData, int width, int height, int numThreads, bool horizontal) {
    totalRows = height;
    std::vector<std::thread> threads;
    int rowsPerThread = height / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int startRow = i * rowsPerThread;
        int endRow = (i + 1) * rowsPerThread;
        if (i == numThreads - 1) {
            endRow = height; // Ensure last thread covers any remaining rows
        }
        threads.emplace_back(mirrorSection, std::ref(imageData), width, height, startRow, endRow, horizontal);
    }

    for (auto& t : threads) {
        t.join();
    }
    std::cout << "\nFinished processing." << std::endl;
}

void rotateImage(const std::vector<unsigned char>& sourceImage, std::vector<unsigned char>& destImage, int originalWidth, int originalHeight, double angle, int& outputWidth, int& outputHeight) {
    calculateRotatedDimensions(originalWidth, originalHeight, angle, outputWidth, outputHeight);
    destImage.resize(outputWidth * outputHeight * 3, 0);

    double rad = angle * (M_PI / 180.0);
    double sinAngle = std::sin(rad);
    double cosAngle = std::cos(rad);

    int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int chunkSize = outputHeight / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        int startY = i * chunkSize;
        int endY = (i == numThreads - 1) ? outputHeight : (i + 1) * chunkSize;
        threads.emplace_back(rotateFunction, std::ref(sourceImage), std::ref(destImage), originalWidth, originalHeight, cosAngle, sinAngle, outputWidth, outputHeight, startY, endY);
    }

    for (auto& thread : threads) {
        thread.join();
    }
}
void resizeImage(const std::vector<unsigned char>& sourceImage, std::vector<unsigned char>& destImage, int originalWidth, int originalHeight, int newWidth, int newHeight) {
    destImage.resize(newWidth * newHeight * 3);
    double scaleX = (double)originalWidth / newWidth;
    double scaleY = (double)originalHeight / newHeight;

    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            int srcX = (int)(x * scaleX);
            int srcY = (int)(y * scaleY);
            Pixel p = getPixel(sourceImage, originalWidth, srcX, srcY);
            int index = (y * newWidth + x) * 3;
            destImage[index] = p.r;
            destImage[index + 1] = p.g;
            destImage[index + 2] = p.b;
        }
    }
}
void cropImage(const std::vector<unsigned char>& sourceImage, std::vector<unsigned char>& destImage, int originalWidth, int originalHeight, int startX, int startY, int newWidth, int newHeight) {
    destImage.resize(newWidth * newHeight * 3);
    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            Pixel p = getPixel(sourceImage, originalWidth, startX + x, startY + y);
            int index = (y * newWidth + x) * 3;
            destImage[index] = p.r;
            destImage[index + 1] = p.g;
            destImage[index + 2] = p.b;
        }
    }
}
void processImageTransformation(const std::vector<std::string>& args) {
    if (args.size() < 4) return; // Ensure basic argument validation

    std::string transformation = args[0];
    std::string inputFilename = args[1];
    std::string outputFilename = args[2];
    int width, height, maxval;
    std::vector<unsigned char> imageData, destImage;

    // Read input image
    std::ifstream inputFile(inputFilename, std::ios::binary);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file: " << inputFilename << std::endl;
        return;
    }
    std::string header;
    inputFile >> header >> width >> height >> maxval;
    inputFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    imageData.resize(width * height * 3);
    inputFile.read(reinterpret_cast<char*>(imageData.data()), imageData.size());
    inputFile.close();

    if (transformation == "mirror") {
        std::string direction = args[3]; // Assuming direction is provided as the fourth argument
        bool horizontal = direction == "horizontal";
        mirrorImage(imageData, width, height, std::thread::hardware_concurrency(), horizontal);
        destImage = imageData; // Mirror operations modify imageData in place
    } else if (transformation == "rotate" && args.size() >= 5) {
        double angle = std::stod(args[4]);
        int outputWidth, outputHeight;
        rotateImage(imageData, destImage, width, height, angle, outputWidth, outputHeight);
        width = outputWidth; // Update width and height to match the output image's dimensions
        height = outputHeight;
    } else if (transformation == "crop" && args.size() >= 7) {
        int startX = std::stoi(args[3]);
        int startY = std::stoi(args[4]);
        int newWidth = std::stoi(args[5]);
        int newHeight = std::stoi(args[6]);
        cropImage(imageData, destImage, width, height, startX, startY, newWidth, newHeight);
        width = newWidth; // Update width and height to match the cropped image's dimensions
        height = newHeight;
    } else if (transformation == "resize" && args.size() >= 5) {
        int newWidth = std::stoi(args[3]);
        int newHeight = std::stoi(args[4]);
        resizeImage(imageData, destImage, width, height, newWidth, newHeight);
        width = newWidth; // Update width and height to match the resized image's dimensions
        height = newHeight;
    } else {
        std::cerr << "Unsupported transformation or insufficient arguments." << std::endl;
        return;
    }

    // Write output image
    std::ofstream outputFile(outputFilename, std::ios::binary);
    if (!outputFile.is_open()) {
        std::cerr << "Error creating output file: " << outputFilename << std::endl;
        return;
    }
    outputFile << "P6\n" << width << " " << height << "\n" << maxval << "\n";
    outputFile.write(reinterpret_cast<const char*>(destImage.data()), destImage.size());
    outputFile.close();
}
void processAllPPMInFolder(const std::string& folderPath, const std::vector<std::string>& transformationArgs) {
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".ppm") {
            std::vector<std::string> args = transformationArgs;
            std::string inputFilename = entry.path().string();
            std::string outputFilename = inputFilename.substr(0, inputFilename.size() - 4) + "_processed.ppm";

            // Modify args to include input and output filenames
            args.insert(args.begin() + 1, outputFilename);
            args.insert(args.begin() + 1, inputFilename);

            processImageTransformation(args);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <folderPath> transformation [transformation parameters...]" << std::endl;
        return 1;
    }

    std::string folderPath = argv[1];
    std::vector<std::string> args(argv + 2, argv + argc); // Exclude the folder path from args

    processAllPPMInFolder(folderPath, args);

    return 0;
}