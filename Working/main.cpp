#include <iostream>
#include <sstream>
#include <vector>
#include <string>

// Assuming the applyTransformation function is defined elsewhere for GPU operations
extern void applyTransformation(const std::vector<std::string>& params);

// Assuming the processImageTransformation function is defined elsewhere for CPU operations
extern void processImageTransformation(const std::vector<std::string>& params);

int main() {
    std::string inputLine;
    std::vector<std::string> params;
    std::string param;

    std::cout << "Enter your command (e.g., mirror input.ppm output.ppm horizontal gpu): ";
    std::getline(std::cin, inputLine); // Read the entire line into inputLine
    std::istringstream iss(inputLine);

    // Split the line into words based on spaces
    while (iss >> param) {
        params.push_back(param);
    }

    if (!params.empty()) {
        std::string processorType = params.back(); // Get the last word which should be "cpu" or "gpu"
        params.pop_back(); // Remove the last element as it's not part of the transformation parameters

        if (processorType == "gpu") {
            applyTransformation(params);
        } else if (processorType == "cpu") {
            processImageTransformation(params);
        } else {
            std::cout << "Error: Last parameter must be 'gpu' or 'cpu' to select the implementation." << std::endl;
        }
    } else {
        std::cout << "Error: No input provided." << std::endl;
    }

    return 0;
}
