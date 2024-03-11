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
#include <cstdlib> // For std::system

namespace fs = std::filesystem;


extern void processAllPPMInFolderCUDA(const std::string& folderPath, const std::vector<std::string>& transformationArgs);
extern void processAllPPMInFolder(const std::string& folderPath, const std::vector<std::string>& transformationArgs);

void convertToPPM(const std::string& inputFilePath, std::string& outputPPMPath) {
    // Using ImageMagick for conversion
    outputPPMPath = inputFilePath + ".ppm"; // Simple conversion for demo
    std::string command = "magick convert " + inputFilePath + " " + outputPPMPath;
    std::system(command.c_str());
}

void convertFromPPM(const std::string& inputPPMPath, const std::string& outputFilePath) {
    // Using ImageMagick for conversion back from PPM
    std::string command = "magick convert " + inputPPMPath + " " + outputFilePath;
    std::system(command.c_str());
}

void processVideoFile(const std::vector<std::string>& params, const std::string& processorType, const std::string& outputFormat) {
    std::string framesDir = "temp_frames";
    std::string audioFile = "temp_audio.aac";
    std::string transformedVideo = params[2]; // Use the output filename provided in params

    // Extract frames and audio using FFmpeg
    std::system(("mkdir -p " + framesDir).c_str());
    std::system(("ffmpeg -i " + params[1] + " " + framesDir + "/frame%04d.ppm").c_str());
    std::system(("ffmpeg -i " + params[1] + " -q:a 0 -map a " + audioFile).c_str());

    // Depending on the processorType, use GPU or CPU to process all PPMs in the folder
    if (processorType == "gpu") {
        processAllPPMInFolderCUDA(framesDir, params);
    } else { // Defaults to CPU processing
        processAllPPMInFolder(framesDir, params);
    }

    // Reassemble video from transformed frames and original audio
    std::string outputFramesPattern = framesDir + "/frame%04d_processed.ppm"; // Adjust based on your actual output naming scheme
    std::system(("ffmpeg -framerate 30 -i " + outputFramesPattern + " -i " + audioFile + " -shortest -c:v libx264 -pix_fmt yuv420p " + transformedVideo).c_str());

    // Cleanup temporary files and directories
    std::system(("rm -rf " + framesDir).c_str());
    std::system(("rm " + audioFile).c_str());
}