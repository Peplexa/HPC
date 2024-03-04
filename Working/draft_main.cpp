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

int main() {
    std::string inputLine;
    std::vector<std::string> params;
    std::string processorType;

    std::cout << "Enter your command (e.g., input.jpg output.jpg mirror horizontal gpu): ";
    std::getline(std::cin, inputLine);

    std::istringstream iss(inputLine);
    std::string param;

    // Split the line into words based on spaces
    while (iss >> param) {
        params.push_back(param);
    }

    if (!params.empty()) {
        processorType = params.back(); // Get the last word which should be "cpu" or "gpu"
        params.pop_back(); // Remove the last element (processor type)

        std::string inputFilePath = params.front();
        params.erase(params.begin()); // Remove input file path from parameters

        std::string outputFilePath = params.front();
        params.erase(params.begin()); // Remove output file path from parameters

        std::string extension = std::filesystem::path(inputFilePath).extension().string();

        std::string ppmInputFile = "intermediate_input.ppm";
        std::string ppmOutputFile = "intermediate_output.ppm";

        // Convert input JPG to PPM if necessary
        if (extension == ".jpg") {
            ConvertJPEGtoPPM(inputFilePath, ppmInputFile);
            inputFilePath = ppmInputFile; // Use the converted PPM file as input
        }

        // Add the PPM input file as the first parameter
        params.insert(params.begin(), ppmOutputFile); // Output file path
        params.insert(params.begin(), inputFilePath); // Input file path

        if (processorType == "gpu") {
            applyTransformation(params);
        } else if (processorType == "cpu") {
            processImageTransformation(params);
        } else {
            std::cout << "Error: Last parameter must be 'gpu' or 'cpu' to select the implementation." << std::endl;
            return 1;
        }

        // Convert the output PPM file back to JPG if necessary
        if (std::filesystem::path(outputFilePath).extension().string() == ".jpg") {
            ConvertPPMtoJPEG(ppmOutputFile, outputFilePath, 75);
        }
    } else {
        std::cout << "Error: No input provided." << std::endl;
    }

    return 0;
}
