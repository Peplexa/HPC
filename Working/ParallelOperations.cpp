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

static void calculateRotatedDimensions(int originalWidth, int originalHeight, double angle, int& outputWidth, int& outputHeight) {
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

//Andy's code here
void convertToGreyscaleSection(std::vector<unsigned char>& imageData, int width, int startRow, int endRow) {
    int rowSize = width * 3; // 3 bytes per pixel (RGB)
    for (int y = startRow; y < endRow; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * rowSize + x * 3;
            unsigned char r = imageData[index];
            unsigned char g = imageData[index + 1];
            unsigned char b = imageData[index + 2];

            // Simple average method for greyscale
            unsigned char grey = static_cast<unsigned char>((r + g + b) / 3);

            imageData[index] = grey;
            imageData[index + 1] = grey;
            imageData[index + 2] = grey;
        }
        completedRows.fetch_add(1);
        updateProgress("Greyscale Conversion");
    }
}

void convertToGreyscale(std::vector<unsigned char>& imageData, int width, int height, int numThreads) {
    totalRows = height;
    std::vector<std::thread> threads;
    int rowsPerThread = height / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int startRow = i * rowsPerThread;
        int endRow = (i + 1) * rowsPerThread;
        if (i == numThreads - 1) {
            endRow = height; // Ensure last thread covers any remaining rows
        }
        threads.emplace_back(convertToGreyscaleSection, std::ref(imageData), width, startRow, endRow);
    }

    for (auto& t : threads) {
        t.join();
    }
    std::cout << "\nFinished greyscale conversion." << std::endl;
}

// Olivia's code here:
// Function to set a pixel in the image data
void setPixel(std::vector<unsigned char>& image, int width, int x, int y, Pixel pixel) { // Pass pixel by value
    int index = (y * width + x) * 3;
    if (index < image.size()) { // Add bounds checking
        image[index] = pixel.r;
        image[index + 1] = pixel.g;
        image[index + 2] = pixel.b;
    }
}

// Function to blend a pixel with opacity
void blendPixel(Pixel& basePixel, const Pixel& watermarkPixel, float opacity) {
    basePixel.r = static_cast<unsigned char>((basePixel.r * (1 - opacity)) + (watermarkPixel.r * opacity));
    basePixel.g = static_cast<unsigned char>((basePixel.g * (1 - opacity)) + (watermarkPixel.g * opacity));
    basePixel.b = static_cast<unsigned char>((basePixel.b * (1 - opacity)) + (watermarkPixel.b * opacity));
}

// Function to blend a section of the watermark onto the base image
void blendWatermarkSection(std::vector<unsigned char>& baseImage, const std::vector<unsigned char>& watermarkImage, int baseWidth, int watermarkWidth, int watermarkHeight, int startX, int startY, int startRow, int endRow, float opacity) {
    for (int y = startRow; y < endRow; ++y) {
        for (int x = 0; x < watermarkWidth; ++x) {
            int baseIndex = ((y + startY) * baseWidth + (x + startX)) * 3;
            int watermarkIndex = (y * watermarkWidth + x) * 3;
            if (baseIndex >= baseImage.size() || watermarkIndex >= watermarkImage.size()) {
                continue; // Skip if we're outside the bounds of either image
            }
            Pixel basePixel = getPixel(baseImage, baseWidth, x + startX, y + startY);
            Pixel watermarkPixel = getPixel(watermarkImage, watermarkWidth, x, y);
            blendPixel(basePixel, watermarkPixel, opacity);
            setPixel(baseImage, baseWidth, x + startX, y + startY, basePixel);
        }
    }
}
bool loadImage(const std::string& filename, std::vector<unsigned char>& imageData, int& width, int& height) {
    std::ifstream inputFile(filename, std::ios::binary);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    std::string header;
    int maxval;

    // Read the header (P6), width, height, and maxval
    inputFile >> header;
    if (header != "P6") {
        std::cerr << "Unsupported image format. Only binary PPM (P6) is supported." << std::endl;
        return false;
    }

    inputFile >> width >> height >> maxval;
    inputFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // Validate image dimensions and maxval
    if (width <= 0 || height <= 0) {
        std::cerr << "Invalid image dimensions." << std::endl;
        return false;
    }
    if (maxval <= 0 || maxval > 255) {
        std::cerr << "Invalid max value. Must be in the range 1-255." << std::endl;
        return false;
    }

    // Resize imageData vector and read pixel data
    imageData.resize(width * height * 3);
    inputFile.read(reinterpret_cast<char*>(imageData.data()), imageData.size());

    if (!inputFile) {
        std::cerr << "Error reading image data." << std::endl;
        return false;
    }

    inputFile.close();
    return true;
}
void superimposeWatermark(std::vector<unsigned char>& baseImage, int baseWidth, int baseHeight, const std::vector<unsigned char>& watermarkImage, int watermarkWidth, int watermarkHeight, float opacity) {
    int startX = (baseWidth - watermarkWidth) / 2;  // Center the watermark
    int startY = (baseHeight - watermarkHeight) / 2;

    for (int y = 0; y < watermarkHeight; ++y) {
        for (int x = 0; x < watermarkWidth; ++x) {
            int baseIndex = ((startY + y) * baseWidth + (startX + x)) * 3;
            int watermarkIndex = (y * watermarkWidth + x) * 3;

            Pixel basePixel = {baseImage[baseIndex], baseImage[baseIndex + 1], baseImage[baseIndex + 2]};
            Pixel watermarkPixel = {watermarkImage[watermarkIndex], watermarkImage[watermarkIndex + 1], watermarkImage[watermarkIndex + 2]};

            // Apply blending based on opacity
            Pixel blendedPixel;
            blendedPixel.r = static_cast<unsigned char>((1 - opacity) * basePixel.r + opacity * watermarkPixel.r);
            blendedPixel.g = static_cast<unsigned char>((1 - opacity) * basePixel.g + opacity * watermarkPixel.g);
            blendedPixel.b = static_cast<unsigned char>((1 - opacity) * basePixel.b + opacity * watermarkPixel.b);

            // Update the base image
            baseImage[baseIndex] = blendedPixel.r;
            baseImage[baseIndex + 1] = blendedPixel.g;
            baseImage[baseIndex + 2] = blendedPixel.b;
        }
    }
}

void processImageTransformation(const std::vector<std::string>& args) {
    // Debug: Print received arguments
    std::cout << "Received arguments:" << std::endl;
    for (const auto& arg : args) {
        std::cout << arg << std::endl;
    }
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
    } else if (transformation == "rotate") {
        double angle = std::stod(args[3]);
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
    } 
    // Inside processImageTransformation, after other transformation conditions
    else if (transformation == "greyscale") {
        convertToGreyscale(imageData, width, height, std::thread::hardware_concurrency());
        destImage = imageData; // Greyscale conversion modifies imageData in place
    }

    else if (transformation == "watermark") {
    if (args.size() < 5) {
        std::cerr << "Insufficient arguments for watermark operation." << std::endl;
        return;
    }

    std::string watermarkFilename = args[3];
    float opacity = std::stof(args[4]) / 100.0f;  // Convert percentage to decimal for opacity
    std::vector<unsigned char> watermarkImage;
    int watermarkWidth, watermarkHeight;

    // Load the watermark image
    if (!loadImage(watermarkFilename, watermarkImage, watermarkWidth, watermarkHeight)) {
        return;
    }

    // Apply the watermark to the base image
    superimposeWatermark(imageData, width, height, watermarkImage, watermarkWidth, watermarkHeight, opacity);

    destImage = imageData;  // Update destImage with the watermarked image
    std::cout << "Watermark applied successfully." << std::endl;
}

    else {
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
/*

int main() {
    
    // Mirror horizontally
    processImageTransformation({"mirror", "input.ppm", "output_mirror_horizontal.ppm", "horizontal"});

    // Mirror vertically
    processImageTransformation({"mirror", "input.ppm", "output_mirror_vertical.ppm", "vertical"});
    
    // Rotate by 90 degrees
    processImageTransformation({"rotate", "input.ppm", "output_rotate_90.ppm", "90"});
    
    // Rotate by 180 degrees
    processImageTransformation({"rotate", "input.ppm", "output_rotate_180.ppm", "180"}); 
    
    // Crop image (example: top-left corner, 100x100 pixels)
    processImageTransformation({"crop", "input.ppm", "output_crop_100x100.ppm", "0", "0", "100", "100"});

    // Resize image to 200x200 pixels
    processImageTransformation({"resize", "input.ppm", "output_resize_200x200.ppm", "200", "200"});

    // Convert to greyscale
    processImageTransformation({"greyscale", "input.ppm", "output_greyscale.ppm"});

    // Apply watermark with 50% opacity and 25% scale
    processImageTransformation({"watermark", "input.ppm", "output_watermark_50_opacity_25_scale.ppm", "watermark.ppm", "50", "25"});
        
    return 0;
}
*/
