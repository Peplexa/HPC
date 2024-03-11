#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

// Assuming the applyTransformation function is defined elsewhere for GPU operations
extern void applyTransformation(const std::vector<std::string>& params);

// Assuming the processImageTransformation function is defined elsewhere for CPU operations
extern void processImageTransformation(const std::vector<std::string>& params);

void processAllPPMInFolder(const std::string& folderPath, const std::vector<std::string>& transformationArgs, const std::string& outputFolderPath, bool useGPU) {
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".ppm") {
            std::vector<std::string> args = transformationArgs;
            std::string inputFilename = entry.path().string();
            std::string outputFilename = outputFolderPath + "/" + entry.path().stem().string() + "_processed.ppm";

            // Modify args to include input and output filenames
            args.insert(args.begin() + 1, outputFilename);
            args.insert(args.begin() + 1, inputFilename);

            if (useGPU) {
                applyTransformation(args);
            } else {
                processImageTransformation(args);
            }
        }
    }
}

bool isVideoFile(const std::string& filename) {
    std::string extension = fs::path(filename).extension().string();
    return extension == ".mp4" || extension == ".avi" || extension == ".mov";
}

int main() {
    std::string inputLine;
    std::vector<std::string> params;
    std::string param;
    
    while (true) {
        params.clear();
        std::cout << "Enter your command, or help for help, or exit to exit: " << std::endl;
        std::cout << "->";
        std::getline(std::cin, inputLine);

        if (inputLine == "exit") {
            break;
        }

        if (inputLine == "help") {
            std::cout << "+------------+---------------------------------------------------------------------------------+\n";
            std::cout << "| Command    | Parameters                                                                      |\n";
            std::cout << "+------------+---------------------------------------------------------------------------------+\n";
            std::cout << "| mirror     | <input_file> <output_file> <direction>                                         |\n";
            std::cout << "| rotate     | <input_file> <output_file> <angle>                                             |\n";
            std::cout << "| resize     | <input_file> <output_file> <width> <height>                                    |\n";
            std::cout << "| crop       | <input_file> <output_file> <start_x> <start_y> <crop_width> <crop_height>      |\n";
            std::cout << "| greyscale  | <input_file> <output_file>                                                     |\n";
            std::cout << "| watermark  | <input_file> <output_file> <watermark_file> <opacity> <scale>                  |\n";
            std::cout << "+------------+---------------------------------------------------------------------------------+\n";
            std::cout << "Note: The last parameter for each command should be either \"gpu\" or \"cpu\" to specify the implementation to use (GPU-based using CUDA or CPU-based using C++).\n";
            std::cout << "\n";
            std::cout << "Example usage:\n";
            std::cout << "- mirror input.jpg output.jpg horizontal gpu\n";
            std::cout << "- mirror input.jpg output.jpg vertical cpu\n";
            std::cout << "- rotate input.jpg output.jpg 90 cpu\n";
            std::cout << "- resize input.jpg output.jpg 800 600 gpu\n";
            std::cout << "- crop input.jpg output.jpg 100 100 500 500 cpu\n";
            std::cout << "- greyscale input.jpg output.jpg gpu\n";
            std::cout << "- watermark input.jpg output.jpg watermark.png 50 25 gpu\n";
            std::cout << "\n";
            continue;
        }

        std::istringstream iss(inputLine);

        while (iss >> param) {
            params.push_back(param);
        }

        if (!params.empty()) {
            std::string processorType = params.back();
            params.pop_back();

            std::string inputFile = params[1];

            if (isVideoFile(inputFile)) {
                // Create temporary folders
                fs::create_directory("temp");
                fs::create_directory("transformed");

                // Open the video file
                cv::VideoCapture video(inputFile);
                if (!video.isOpened()) {
                    std::cout << "Error: Failed to open video file." << std::endl;
                    continue;
                }

                // Get the video properties
                int frameWidth = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
                int frameHeight = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
                int frameCount = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));
                double fps = video.get(cv::CAP_PROP_FPS);

                // Start the timer
                auto startTime = std::chrono::high_resolution_clock::now();

                // Extract frames from the video and save them as PPM files in the "temp" folder
                int frameNumber = 0;
                while (true) {
                    cv::Mat frame;
                    video >> frame;
                    if (frame.empty())
                        break;

                    std::string frameName = "temp/frame_" + std::to_string(frameNumber) + ".ppm";
                    cv::imwrite(frameName, frame);
                    frameNumber++;
                }

                // Process all PPM files in the "temp" folder and save the transformed files in the "transformed" folder
                bool useGPU = (processorType == "gpu");
                
                if (params[0] == "rotate") {
                    double angle = std::stod(params[3]);
                    for (int i = 0; i < frameNumber; i++) {
                        std::string inputFrameName = "temp/frame_" + std::to_string(i) + ".ppm";
                        std::string outputFrameName = "transformed/frame_" + std::to_string(i) + "_processed.ppm";
                        
                        // Read the input frame
                        cv::Mat inputFrame = cv::imread(inputFrameName);
                        
                        // Perform rotation using OpenCV
                        cv::Mat outputFrame;
                        cv::Point2f center(inputFrame.cols / 2.0, inputFrame.rows / 2.0);
                        cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);
                        cv::warpAffine(inputFrame, outputFrame, rotationMatrix, inputFrame.size());
                        
                        // Save the rotated frame
                        cv::imwrite(outputFrameName, outputFrame);
                    }
                } else {
                    processAllPPMInFolder("temp", params, "transformed", useGPU);
                }

                // Create the output video file
                std::string outputFile = params[2];
                cv::VideoWriter outputVideo(outputFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frameWidth, frameHeight));

                // Write the transformed frames to the output video
                for (int i = 0; i < frameNumber; i++) {
                    std::string transformedFrameName = "transformed/frame_" + std::to_string(i) + "_processed.ppm";
                    cv::Mat transformedFrame = cv::imread(transformedFrameName);
                    outputVideo.write(transformedFrame);
                }

                // Release the video writer
                outputVideo.release();

                // Stop the timer and calculate the processing time
                auto endTime = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
                std::cout << "Processing time: " << duration.count() << " microseconds" << std::endl;

                // Clean up temporary folders and files
                fs::remove_all("temp");
                fs::remove_all("transformed");
            } else {
                // Start the timer
                auto startTime = std::chrono::high_resolution_clock::now();

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

                // Stop the timer and calculate the processing time
                auto endTime = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
                std::cout << "Processing time: " << duration.count() << " microseconds, completed successfully" << std::endl;
            }
        } else {
            std::cout << "Error: No input provided." << std::endl;
        }
    }

    return 0;
}