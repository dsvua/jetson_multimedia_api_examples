
#include "rawImageSaver.h"
#include <fstream>


void rawImageSaver(const unsigned int width, const unsigned int height, uint8_t *pixels, const char* filename){

    FILE *file = fopen(filename, "wb");

    if (file)
    {
        setbuf(file, NULL);
        fwrite(pixels, sizeof(uint8_t), width*height*sizeof(uint8_t), file);
        fflush(file);
        fclose(file);

        // to convert binary file to png for human review:
        // convert -size 1280x1440 -depth 8 gray:output000.bin output000.png
    }
}