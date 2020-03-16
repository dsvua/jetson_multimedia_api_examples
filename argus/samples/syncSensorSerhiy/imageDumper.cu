#include "histogram.h"
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include "rawImageSaver.h"
#include <iostream>


__global__ void cudaDumpImage(CUsurfObject surface_left, CUsurfObject surface_right, uint8_t* d_image, unsigned int width, unsigned int height){
    // global position and size
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x<width && y < height) {
        d_image[y*width+x]              = surf2Dread<uint8_t>(surface_left, x, y);
        d_image[y*width+x+height*width] = surf2Dread<uint8_t>(surface_right, x, y);

        // printing for debugging purpose, it should pring any number but not 0
        if (x == 1000 && y == 300){
            printf("surf2Dread at 1000x300 left: %3u, right: %3u \n", surf2Dread<uint8_t>(surface_left, x, y), surf2Dread<uint8_t>(surface_right, x, y));
        }
    }
}

void checkCudaError(cudaError_t err, const char * message){
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to %s (error code %03u %s)!\n",
                         message, err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    
}

void imageDumper(CUsurfObject surface_left, CUsurfObject surface_right, unsigned int width, unsigned int height, unsigned int frameNumber)
{
    // Set flag to enable zero copy access
    cudaError_t err = cudaSuccess;

    dim3 block(8, 8);
    dim3 grid(width/block.x+1,height/block.y+1);

    printf("Picture size is %u x %u \n", width, height);
    printf("cudaMallocManaged image memory\n");
    uint8_t *image, *d_image;
    size_t mallocSize = width*height*sizeof(uint8_t)*2;
    image = (uint8_t*)malloc(mallocSize); // multiply by 2 as we have 2 images
    // err = cudaMallocManaged(&image,width*height*sizeof(uint8_t));
    // checkCudaError(err, "cudaMallocManaged");
 
    err = cudaMalloc(&d_image, mallocSize);
    checkCudaError(err, "cudaMalloc");

    printf("Calling cudaDumpImage kernel...\n");
    cudaDumpImage<<<grid, block>>>(surface_left, surface_right, d_image, width, height);
    err = cudaGetLastError();
    checkCudaError(err, "cudaDumpImage kernel");

    err = cudaMemcpy(image, d_image, mallocSize, cudaMemcpyDeviceToHost);
    checkCudaError(err, "cudaMemcpy");
    // for(int i = 13120; i < 13320; i++) printf("results[%1u] %3u \n", i, image[i]);
    printf("results[%1u] %3u \n", 13132, image[13132]);

    err = cudaDeviceSynchronize();
    checkCudaError(err, "cudaDeviceSynchronize");

    printf("Dumping frame to image %d \n", frameNumber);

    char filename_bin[256];
    sprintf(filename_bin, "output%03u.bin", frameNumber);
    rawImageSaver(width, height*2, image, filename_bin); // height*2 as we have two images

    printf("Freeing d_image memory\n");
    cudaFree(d_image);
    printf("Freeing image memory\n");
    delete image;
}
