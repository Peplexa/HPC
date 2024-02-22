#include <iostream>
#include <vector>
#include <thread>
#include <jpeglib.h>
#include <stdio.h>

void ConvertPPMtoJPEG(const std::string& inputPath, const std::string& outputPath, int quality = 75) {
    // Open PPM file
    FILE* infile = fopen(inputPath.c_str(), "rb");
    if (!infile) {
        std::cerr << "Cannot open " << inputPath << std::endl;
        return;
    }

    // Read PPM header (P6 format)
    int width, height, maxval;
    fscanf(infile, "P6 %d %d %d%*c", &width, &height, &maxval);
    unsigned char* raw_image = new unsigned char[width * height * 3];
    fread(raw_image, width * height * 3, 1, infile);
    fclose(infile);

    // Initialize libjpeg structures
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    JSAMPROW row_pointer[1];
    FILE* outfile = fopen(outputPath.c_str(), "wb");
    if (!outfile) {
        std::cerr << "Cannot open " << outputPath << std::endl;
        delete[] raw_image;
        return;
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3; // RGB
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = &raw_image[cinfo.next_scanline * width * 3];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    // Cleanup
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
    delete[] raw_image;
}

void ProcessImages(const std::vector<std::pair<std::string, std::string>>& images, int quality = 75) {
    std::vector<std::thread> threads;

    // Create a thread for each image conversion
    for (const auto& img : images) {
        threads.emplace_back(ConvertPPMtoJPEG, img.first, img.second, quality);
    }

    // Wait for all threads to finish
    for (auto& thread : threads) {
        thread.join();
    }
}

int main() {
    // Example usage: Convert two images in parallel
    std::vector<std::pair<std::string, std::string>> images = {
        {"input1.ppm", "output1.jpg"},
        {"input2.ppm", "output2.jpg"}
    };

    ProcessImages(images);

    return 0;
}
