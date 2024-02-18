#include "superimpose.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <jpeglib.h>

void saveImage(const char* filename, unsigned char* buffer, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    // Write only RGB components (ignoring Alpha channel)
    for (int i = 0; i < width * height; ++i) {
        file.write(reinterpret_cast<char*>(&buffer[i * 4]), 3);
    }
    file.close();
}

void write_JPEG_file(const char *filename, int quality, unsigned char* image_buffer, int width, int height) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    FILE *outfile;
    JSAMPROW row_pointer[1];
    int row_stride;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    if ((outfile = fopen(filename, "wb")) == NULL) {
        fprintf(stderr, "can't open %s\n", filename);
        exit(1);
    }
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3; // Only RGB components
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);

    jpeg_start_compress(&cinfo, TRUE);

    row_stride = width * 3; // 3 bytes per pixel (RGB)

    // Writing RGB components
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = &image_buffer[cinfo.next_scanline * row_stride];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
}

int main() {
    const int largeWidth = 1024;
    const int largeHeight = 768;
    const int smallWidth = 256;
    const int smallHeight = 256;
    const int startX = 384; // Position to start overlaying the small image
    const int startY = 256;

    // Allocate memory for large and small images
    unsigned char* largeImage = new unsigned char[largeWidth * largeHeight * 4];
    unsigned char* smallImage = new unsigned char[smallWidth * smallHeight * 4];

    // Initialize the large image with a checkerboard pattern
    for (int y = 0; y < largeHeight; ++y) {
        for (int x = 0; x < largeWidth; ++x) {
            int index = (y * largeWidth + x) * 4;
            bool isWhite = (x / 100 + y / 100) % 2 == 0;
            largeImage[index + 0] = isWhite ? 255 : 0; // Red
            largeImage[index + 1] = isWhite ? 255 : 0; // Green
            largeImage[index + 2] = isWhite ? 255 : 0; // Blue
            largeImage[index + 3] = 255;               // Alpha, not used for PPM
        }
    }

    // Initialize the small image with a solid color
    for (int i = 0; i < smallWidth * smallHeight; ++i) {
        smallImage[i * 4 + 0] = 255; // Red
        smallImage[i * 4 + 1] = 0;   // Green
        smallImage[i * 4 + 2] = 0;   // Blue
        smallImage[i * 4 + 3] = 255; // Alpha, not used for PPM
    }

    // Overlay the small image onto the large image
    overlayImages(largeImage, smallImage, largeWidth, largeHeight, smallWidth, smallHeight, startX, startY);

    // Save the result to a PPM file
    saveImage("overlay_output.ppm", largeImage, largeWidth, largeHeight);

    // Convert the PPM to a JPEG file
    write_JPEG_file("overlay_output.jpg", 95, largeImage, largeWidth, largeHeight);

    // Free the allocated memory
    delete[] largeImage;
    delete[] smallImage;

    return 0;
}
