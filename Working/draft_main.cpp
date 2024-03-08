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

    // Step 1: Take input JPEG
    std::cout << "Enter the path to the input JPEG file: ";
    std::getline(std::cin, inputJpgPath);

    // Step 2: Convert JPEG to PPM
    ConvertJPEGtoPPM(inputJpgPath, outputPpmPath);

    // Step 3: Feed PPM information to the transformation program
    // Example: mirror output.ppm output_transformed.ppm horizontal cpu
    std::string processorType;
    std::cout << "Enter your command (e.g., mirror output.ppm output_transformed.ppm horizontal cpu): ";
    std::string inputLine;
    std::getline(std::cin, inputLine); // Read the entire line into inputLine
    std::istringstream iss(inputLine);

    // Split the line into words based on spaces
    std::string param;
    while (iss >> param) {
        params.push_back(param);
    }

    if (!params.empty()) {
        processorType = params.back(); // Get the last word which should be "cpu" or "gpu"
        params.pop_back(); // Remove the last element as it's not part of the transformation parameters
    } else {
        std::cerr << "Error: No transformation command provided." << std::endl;
        return 1;
    }

    // Step 4: Program outputs .ppm transformation
    if (processorType == "gpu") {
        applyTransformation(params);
    } else if (processorType == "cpu") {
        processImageTransformation(params);
    } else {
        std::cerr << "Error: Last parameter must be 'gpu' or 'cpu' to select the implementation." << std::endl;
        return 1;
    }
    std::string transformedPpmPath = "output_transformed.ppm"; // This should be the actual path to the mirrored image

    // convert transformed .ppm file back to JPEG 
    ConvertPPMtoJPEG(transformedPpmPath, outputJpgPath, 75);
    std::cout << "Output JPEG file generated: " << outputJpgPath << std::endl;
    return 0;
}
