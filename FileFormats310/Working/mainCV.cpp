#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// Assuming the applyTransformation function is defined elsewhere for GPU operations
extern void applyTransformation(const std::vector<std::string>& params);

// Assuming the processImageTransformation function is defined elsewhere for CPU operations
extern void processImageTransformation(const std::vector<std::string>& params);

extern void processAllImagesInFolder(const std::string& folderPath, const std::vector<std::string>& transformationArgs);

extern void processAllImagesInFolderCUDA(const std::string& folderPath, const std::vector<std::string>& transformationArgs);

int main() {
    std::string inputLine;
    std::vector<std::string> params;
    std::string param;
    
    while (true) {
        params.clear();
        std::cout << "Enter your command (e.g., watermark input.jpg output.jpg [watermark.png] [opacity] [scale] gpu) or type 'exit' to quit: ";
        std::getline(std::cin, inputLine);

        if (inputLine == "exit") {
            break;
        }

        std::istringstream iss(inputLine);

        while (iss >> param) {
            params.push_back(param);
        }

        if (!params.empty()) {
            std::string processorType = params.back();
            params.pop_back();

            std::string inputFile = params[1];
            std::string outputFile = params[2];

            cv::Mat inputImage = cv::imread(inputFile, cv::IMREAD_COLOR);

            if (inputImage.empty()) {
                std::cout << "Error: Failed to open input image file." << std::endl;
                continue;
            }

            params[1] = "input.ppm";
            cv::imwrite("input.ppm", inputImage);

            if (params.size() >= 6 && params[0] == "watermark") {
                std::string watermarkFile = params[3];
                cv::Mat watermarkImage = cv::imread(watermarkFile, cv::IMREAD_COLOR);

                if (watermarkImage.empty()) {
                    std::cout << "Error: Failed to open watermark image file." << std::endl;
                    continue;
                }

                params[3] = "watermark.ppm";
                cv::imwrite("watermark.ppm", watermarkImage);
            }

            if (processorType == "gpu") {
                applyTransformation(params);
            } else if (processorType == "cpu") {
                processImageTransformation(params);
            } else {
                std::cout << "Error: Last parameter must be 'gpu' or 'cpu' to select the implementation." << std::endl;
                continue;
            }

            cv::Mat outputImage = cv::imread(outputFile, cv::IMREAD_COLOR);
            cv::imwrite(outputFile, outputImage);

            std::remove("input.ppm");
            if (params.size() >= 6 && params[0] == "watermark") {
                std::remove("watermark.ppm");
            }
        } else {
            std::cout << "Error: No input provided." << std::endl;
        }
    }

    return 0;
}