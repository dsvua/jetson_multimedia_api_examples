#ifndef RAWIMAGESAVER_H
#define RAWIMAGESAVER_H

#include <stdio.h>
#include <stdint.h>

void rawImageSaver(const unsigned int width, const unsigned int height, uint8_t *pixels, const char* filename);

#endif