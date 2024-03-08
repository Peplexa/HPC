#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>

// Assuming the applyTransformation function is defined elsewhere for GPU operations
extern void applyTransformation(const std::vector<std::string>& params);

// Assuming the processImageTransformation function is defined elsewhere for CPU operations
extern void processImageTransformation(const std::vector<std::string>& params);

void ConvertJPEGtoPPM(const std::string& inputPath, const std::string& outputPath);

void ConvertPPMtoJPEG(const std::string& inputPath, const std::string& outputPath, int quality = 75);
namespace fs = std::filesystem;

int main() {
    std::string inputJpgPath;
    std::string outputPpmPath = "output.ppm"; // Default output PPM file path
    std::string outputJpgPath = "output.jpg"; // Default output JPEG file path
    std::vector<std::string> params;

      // Step 1: Ask for the type of operation
    std::cout << "Enter the type of operation (mirror, rotate, greyscale, watermark): ";
    std::string operationType;
    std::getline(std::cin, operationType);

    // Step 2: Take input JPEG
    std::cout << "Enter the path to the input JPEG file: ";
    std::getline(std::cin, inputJpgPath);

    // Step 3: Convert JPEG to PPM
    ConvertJPEGtoPPM(inputJpgPath, outputPpmPath);

    // Step 4: Get additional parameters based on the operation
    if (operationType == "mirror") {
        std::string direction;
        std::cout << "Enter mirror direction (horizontal or vertical): ";
        std::cin >> direction;
        params = {operationType, outputPpmPath, "output_transformed.ppm", direction};
    } else if (operationType == "rotate") {
        std::string angle;
        std::cout << "Enter rotation angle in degrees (e.g., 90): ";
        std::cin >> angle;
        params = {operationType, outputPpmPath, "output_transformed.ppm", angle};
    } else if (operationType == "greyscale") {
        params = {operationType, outputPpmPath, "output_transformed.ppm"};
    } else if (operationType == "watermark") {
    std::string watermarkPath, opacity;
    std::cout << "Enter the path to the watermark image: ";
    std::cin >> watermarkPath;

    // Convert watermark image to PPM if it's not already in that format
    std::string watermarkPpmPath = "watermark.ppm"; // Temporary PPM filename for watermark
    if (watermarkPath.find(".jpg") != std::string::npos || watermarkPath.find(".jpeg") != std::string::npos) {
        ConvertJPEGtoPPM(watermarkPath, watermarkPpmPath);
        watermarkPath = watermarkPpmPath; // Use the converted PPM file as the watermark
    }

    std::cout << "Enter the opacity for the watermark (0-100): ";
    std::cin >> opacity;
    params = {operationType, outputPpmPath, "output_transformed.ppm", watermarkPath, opacity};
    } else {
        std::cerr << "Unsupported operation type." << std::endl;
        return 1;
    }

    // Step 5: Apply the transformation
    processImageTransformation(params);

    // Step 6: Convert the transformed PPM back to JPEG
    ConvertPPMtoJPEG("output_transformed.ppm", outputJpgPath, 75);
    std::cout << "Output JPEG file generated: " << outputJpgPath << std::endl;

    return 0;
}
  
