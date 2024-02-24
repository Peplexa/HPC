#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <omp.h>

struct Pixel {
    unsigned char r, g, b;
};

void overlayImagesParallel(std::vector<unsigned char>& baseImage, const std::vector<unsigned char>& overlayImage, int baseWidth, int baseHeight, int overlayWidth, int overlayHeight, int posX, int posY) {
    #pragma omp parallel for collapse(2)
    for (int y = posY; y < posY + overlayHeight; ++y) {
        for (int x = posX; x < posX + overlayWidth; ++x) {
            if (x < baseWidth && y < baseHeight) {
                int baseIdx = (y * baseWidth + x) * 3;
                int overlayIdx = ((y - posY) * overlayWidth + (x - posX)) * 3;

                baseImage[baseIdx] = overlayImage[overlayIdx];
                baseImage[baseIdx + 1] = overlayImage[overlayIdx + 1];
                baseImage[baseIdx + 2] = overlayImage[overlayIdx + 2];
            }
        }
    }
}

void processImageTransformation(const std::vector<std::string>& args) {
    if (args.size() < 5) {  // Ensure enough arguments for overlay transformation
        std::cerr << "Insufficient arguments provided." << std::endl;
        return;
    }

    std::string transformation = args[0];
    std::string baseImageFilename = args[1];
    std::string overlayImageFilename = args[2];
    std::string outputFilename = args[3];
    int posX = std::stoi(args[4]);
    int posY = std::stoi(args[5]);
    int width, height, overlayWidth, overlayHeight, maxval;
    std::vector<unsigned char> baseImage, overlayImage;

    // Load the base image
    loadPPM(baseImageFilename, baseImage, width, height, maxval);

    // Load the overlay image
    loadPPM(overlayImageFilename, overlayImage, overlayWidth, overlayHeight, maxval);

    if (transformation == "overlay") {
        // Perform the overlay operation
        overlayImagesParallel(baseImage, overlayImage, width, height, overlayWidth, overlayHeight, posX, posY);
    } else {
        std::cerr << "Unsupported transformation: " << transformation << std::endl;
        return;
    }
    // Save the output image
    savePPM(outputFilename, baseImage, width, height, maxval);
}

void loadPPM(const std::string& filename, std::vector<unsigned char>& image, int& width, int& height, int& maxval) {
    std::ifstream inputFile(filename, std::ios::binary);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    std::string header;
    inputFile >> header;
    if (header != "P6") {
        std::cerr << "Unsupported format or not a P6 PPM file." << std::endl;
        inputFile.close();
        exit(1);
    }

    inputFile >> width >> height >> maxval;
    inputFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Skip to the end of the header line

    image.resize(width * height * 3);
    inputFile.read(reinterpret_cast<char*>(image.data()), image.size());
    inputFile.close();
}

void savePPM(const std::string& filename, const std::vector<unsigned char>& image, int width, int height, int maxval) {
    std::ofstream outputFile(filename, std::ios::binary);
    if (!outputFile.is_open()) {
        std::cerr << "Error creating output file: " << filename << std::endl;
        exit(1);
    }

    outputFile << "P6\n" << width << " " << height << "\n" << maxval << "\n";
    outputFile.write(reinterpret_cast<const char*>(image.data()), image.size());
    outputFile.close();
}
