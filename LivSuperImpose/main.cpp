#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <omp.h>

void loadPPM(const std::string& filename, unsigned char** image, int* width, int* height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        exit(1);
    }

    std::string line;
    getline(file, line);
    getline(file, line); 
    while (line[0] == '#') {
        getline(file, line);
    }
    std::istringstream dimensions(line);

    dimensions >> *width >> *height;

    getline(file, line); // Max color value
    *image = new unsigned char[(*width) * (*height) * 3];
    file.read(reinterpret_cast<char*>(*image), (*width) * (*height) * 3);
    file.close();
}

void savePPM(const std::string& filename, unsigned char* image, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        exit(1);
    }

    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<char*>(image), width * height * 3);
    file.close();
}

void overlayImagesParallel(const std::vector<std::string>& params) {
    // Assuming params are: largeImagePath, smallImagePath, outputImagePath, startX, startY
    std::string largeImagePath = params[0];
    std::string smallImagePath = params[1];
    std::string outputImagePath = params[2];
    int startX = std::stoi(params[3]);
    int startY = std::stoi(params[4]);

    unsigned char *largeImage, *smallImage;
    int largeWidth, largeHeight, smallWidth, smallHeight;

    // Load the images
    loadPPM(largeImagePath, &largeImage, &largeWidth, &largeHeight);
    loadPPM(smallImagePath, &smallImage, &smallWidth, &smallHeight);

    // Perform the overlay in parallel
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < smallHeight; ++y) {
        for (int x = 0; x < smallWidth; ++x) {
            if (y + startY < largeHeight && x + startX < largeWidth) {
                int largeIdx = ((y + startY) * largeWidth + (x + startX)) * 3;
                int smallIdx = (y * smallWidth + x) * 3;
                for (int c = 0; c < 3; ++c) {
                    largeImage[largeIdx + c] = smallImage[smallIdx + c];
                }
            }
        }
    }

    // Save the result
    savePPM(outputImagePath, largeImage, largeWidth, largeHeight);

    delete[] largeImage;
    delete[] smallImage;
}


