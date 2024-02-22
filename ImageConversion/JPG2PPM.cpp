#include <iostream>
#include <vector>
#include <thread>
#include <jpeglib.h>
#include <stdio.h>

void ConvertJPEGtoPPM(const std::string& inputPath, const std::string& outputPath) {
    // Open JPEG file
    FILE* infile = fopen(inputPath.c_str(), "rb");
    if (!infile) {
        std::cerr << "Cannot open " << inputPath << std::endl;
        return;
    }

    // Initialize libjpeg structures
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    // Write to PPM
    FILE* outfile = fopen(outputPath.c_str(), "wb");
    if (!outfile) {
        std::cerr << "Cannot open " << outputPath << std::endl;
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return;
    }

    fprintf(outfile, "P6\n%d %d\n255\n", cinfo.output_width, cinfo.output_height);
    unsigned char* row_buffer = (unsigned char*)malloc(cinfo.output_width * cinfo.num_components);
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, &row_buffer, 1);
        fwrite(row_buffer, cinfo.output_width * cinfo.num_components, 1, outfile);
    }
    free(row_buffer);

    // Cleanup
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    fclose(outfile);
}

void ProcessImages(const std::vector<std::pair<std::string, std::string>>& images) {
    std::vector<std::thread> threads;

    // Create a thread for each image conversion
    for (const auto& img : images) {
        threads.emplace_back(ConvertJPEGtoPPM, img.first, img.second);
    }

    // Wait for all threads to finish
    for (auto& thread : threads) {
        thread.join();
    }
}

int main() {
    // Example usage: Convert two images in parallel
    std::vector<std::pair<std::string, std::string>> images = {
        {"input1.jpg", "output1.ppm"},
        {"input2.jpg", "output2.ppm"}
    };

    ProcessImages(images);

    return 0;
}
