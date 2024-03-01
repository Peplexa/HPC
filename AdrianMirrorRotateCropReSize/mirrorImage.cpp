#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>

std::mutex outputMutex;
std::atomic<int> completedRows(0);
int totalRows;

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

bool getUserChoice(const std::string& prompt) {
    std::string response;
    std::cout << prompt;
    std::getline(std::cin, response);
    return response[0] == 'Y' || response[0] == 'y';
}

int main() {
    std::string inputFilename = "input.ppm";
    std::ifstream file(inputFilename, std::ios::binary);
    std::string outputFilename = "mirrored.ppm";

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << inputFilename << std::endl;
        return 1;
    }

    std::string line;
    std::getline(file, line); // Read the magic number
    if (line != "P6") {
        std::cerr << "Unsupported format or not a P6 PPM file." << std::endl;
        return 1;
    }

    int width, height, maxval;
    file >> width >> height >> maxval;
    file.ignore(); // Skip the single whitespace after the maxval

    std::vector<unsigned char> imageData(width * height * 3); // 3 bytes per pixel
    file.read(reinterpret_cast<char*>(imageData.data()), imageData.size());
    file.close();

    bool horizontalMirror = getUserChoice("Horizontal Mirror (Y/N)? ");
    bool verticalMirror = getUserChoice("Vertical Mirror (Y/N)? ");

    if (horizontalMirror) {
        mirrorImage(imageData, width, height, 4, true);
    }
    if (verticalMirror) {
        completedRows = 0; // Reset for the next operation
        mirrorImage(imageData, width, height, 4, false);
    }

    std::ofstream outFile(outputFilename, std::ios::binary);
    outFile << "P6\n" << width << " " << height << "\n" << maxval << "\n";
    outFile.write(reinterpret_cast<const char*>(imageData.data()), imageData.size());
    outFile.close();

    return 0;
}
