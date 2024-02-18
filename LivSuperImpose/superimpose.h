#ifndef SUPERIMPOSE_H
#define SUPERIMPOSE_H

#ifdef __cplusplus
extern "C" {
#endif

void overlayImages(unsigned char* largeImage, unsigned char* smallImage, int largeWidth, int largeHeight, int smallWidth, int smallHeight, int startX, int startY);

#ifdef __cplusplus
}
#endif

#endif // SUPERIMPOSE_H
