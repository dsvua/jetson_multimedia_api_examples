/*
 * Copyright (c) 2016 - 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Argus/Argus.h>
#include <cuda.h>
#include <cudaEGL.h>

#include "CommonOptions.h"
#include "CUDAHelper.h"
#include "EGLGlobal.h"
#include "Error.h"
#include "KLDistance.h"
#include "Thread.h"

#include <chrono> 
#include <iostream> 

// Histogram generation methods borrowed from cudaHistogram sample.
#include "imageDumper.h"

using namespace Argus;

namespace ArgusSamples
{

/*
 * This sample opens a session with two sensors, it then using CUDA computes the histogram
 * of the sensors and computes a KL distance between the two histograms. A small value near
 * 0 indicates that the two images are alike. The processing of the images happens in the worker
 * thread of StereoDisparityConsumerThread. While the main app thread is used to drive the captures.
 */

// Constants.
static const Size2D<uint32_t> STREAM_SIZE(1280, 720);

// Globals and derived constants.
EGLDisplayHolder g_display;

// Debug print macros.
#define PRODUCER_PRINT(...) printf("PRODUCER: " __VA_ARGS__)
#define CONSUMER_PRINT(...) printf("CONSUMER: " __VA_ARGS__)

/*******************************************************************************
 * Argus disparity class
 *   This class will analyze frames from 2 synchronized sensors and compute the
 *   KL distance between the two images. Large values of KL indicate a large disparity
 *   while a value of 0.0 indicates that the images are alike.
 ******************************************************************************/
class StereoDisparityConsumerThread : public Thread
{
public:
    explicit StereoDisparityConsumerThread(IEGLOutputStream *leftStream,
                                           IEGLOutputStream *rightStream)
                                         : m_leftStream(leftStream)
                                         , m_rightStream(rightStream)
                                         , m_cudaContext(0)
                                         , m_cuStreamLeft(NULL)
                                         , m_cuStreamRight(NULL)
    {
    }
    ~StereoDisparityConsumerThread()
    {
    }

private:
    /** @name Thread methods */
    /**@{*/
    virtual bool threadInitialize();
    virtual bool threadExecute();
    virtual bool threadShutdown();
    /**@}*/

    IEGLOutputStream *m_leftStream;
    IEGLOutputStream *m_rightStream;
    CUcontext         m_cudaContext;
    CUeglStreamConnection m_cuStreamLeft;
    CUeglStreamConnection m_cuStreamRight;
    int frameNumber;
};

/**
 * Utility class to acquire and process an EGLStream frame from a CUDA
 * consumer as long as the object is in scope.
 */
class ScopedCudaEGLStreamFrameAcquire
{
public:
    /**
     * Constructor blocks until a frame is acquired or an error occurs (ie. stream is
     * disconnected). Caller should check which condition was satisfied using hasValidFrame().
     */
    ScopedCudaEGLStreamFrameAcquire(CUeglStreamConnection& connection_left, 
            CUeglStreamConnection& connection_right);

    /**
     * Destructor releases frame back to EGLStream.
     */
    ~ScopedCudaEGLStreamFrameAcquire();

    /**
     * Returns true if a frame was acquired (and is compatible with this consumer).
     */
    bool hasValidFrame() const;

    /**
     * Use CUDA to generate a histogram from the acquired frame.
     * @param[out] histogramData Output array for the histogram.
     * @param[out] time Time to generate histogram, in milliseconds.
     */
    // bool generateHistogram(unsigned int histogramData[HISTOGRAM_BINS], float *time);
    bool dumpImage(int frameNumber);

    /**
     * Returns the size (resolution) of the frame.
     */
    Size2D<uint32_t> getSize() const;

private:

    /**
     * Returns whether or not the frame format is supported.
     */
    bool checkFormat() const;

    CUeglStreamConnection& m_connection_left;
    CUstream m_stream_left;
    CUgraphicsResource m_resource_left;
    CUeglFrame m_frame_left;

    CUeglStreamConnection& m_connection_right;
    CUstream m_stream_right;
    CUgraphicsResource m_resource_right;
    CUeglFrame m_frame_right;
};

bool StereoDisparityConsumerThread::threadInitialize()
{
    // Create CUDA and connect egl streams.
    PROPAGATE_ERROR(initCUDA(&m_cudaContext));

    CONSUMER_PRINT("Connecting CUDA consumer to left stream\n");
    CUresult cuResult = cuEGLStreamConsumerConnect(&m_cuStreamLeft, m_leftStream->getEGLStream());
    if (cuResult != CUDA_SUCCESS)
    {
        ORIGINATE_ERROR("Unable to connect CUDA to EGLStream as a consumer (CUresult %s)",
            getCudaErrorString(cuResult));
    }

    CONSUMER_PRINT("Connecting CUDA consumer to right stream\n");
    cuResult = cuEGLStreamConsumerConnect(&m_cuStreamRight, m_rightStream->getEGLStream());
    if (cuResult != CUDA_SUCCESS)
    {
        ORIGINATE_ERROR("Unable to connect CUDA to EGLStream as a consumer (CUresult %s)",
            getCudaErrorString(cuResult));
    }
    return true;
}

bool StereoDisparityConsumerThread::threadExecute()
{
    CONSUMER_PRINT("Waiting for Argus producer to connect to left stream.\n");
    m_leftStream->waitUntilConnected();

    CONSUMER_PRINT("Waiting for Argus producer to connect to right stream.\n");
    m_rightStream->waitUntilConnected();

    CONSUMER_PRINT("Streams connected, processing frames.\n");
    // unsigned int histogramLeft[HISTOGRAM_BINS];
    // unsigned int histogramRight[HISTOGRAM_BINS];
    while (true)
    {
        EGLint streamState = EGL_STREAM_STATE_CONNECTING_KHR;

        // Check both the streams and proceed only if they are not in DISCONNECTED state.
        if (!eglQueryStreamKHR(
                    m_leftStream->getEGLDisplay(),
                    m_leftStream->getEGLStream(),
                    EGL_STREAM_STATE_KHR,
                    &streamState) || (streamState == EGL_STREAM_STATE_DISCONNECTED_KHR))
        {
            CONSUMER_PRINT("left : EGL_STREAM_STATE_DISCONNECTED_KHR received\n");
            break;
        }

        if (!eglQueryStreamKHR(
                    m_rightStream->getEGLDisplay(),
                    m_rightStream->getEGLStream(),
                    EGL_STREAM_STATE_KHR,
                    &streamState) || (streamState == EGL_STREAM_STATE_DISCONNECTED_KHR))
        {
            CONSUMER_PRINT("right : EGL_STREAM_STATE_DISCONNECTED_KHR received\n");
            break;
        }

        ScopedCudaEGLStreamFrameAcquire left_and_right(m_cuStreamLeft, m_cuStreamRight);

        if (!left_and_right.hasValidFrame())
            break;

        left_and_right.dumpImage(frameNumber);
        frameNumber++;
    }
    CONSUMER_PRINT("No more frames. Cleaning up.\n");

    PROPAGATE_ERROR(requestShutdown());

    return true;
}

bool StereoDisparityConsumerThread::threadShutdown()
{
    // Disconnect from the streams.
    cuEGLStreamConsumerDisconnect(&m_cuStreamLeft);
    cuEGLStreamConsumerDisconnect(&m_cuStreamRight);

    PROPAGATE_ERROR(cleanupCUDA(&m_cudaContext));

    CONSUMER_PRINT("Done.\n");
    return true;
}

ScopedCudaEGLStreamFrameAcquire::ScopedCudaEGLStreamFrameAcquire(CUeglStreamConnection& connection_left, 
    CUeglStreamConnection& connection_right)
    : m_connection_left(connection_left)
    , m_connection_right(connection_right)
    , m_stream_left(NULL)
    , m_stream_right(NULL)
    , m_resource_left(0)
    , m_resource_right(0)
{
    CUresult r = cuEGLStreamConsumerAcquireFrame(&m_connection_left, &m_resource_left, &m_stream_left, -1);
    if (r == CUDA_SUCCESS)
    {
        cuGraphicsResourceGetMappedEglFrame(&m_frame_left, m_resource_left, 0, 0);
    } else {
        printf ("Failed to acquire left image");
    }
    r = cuEGLStreamConsumerAcquireFrame(&m_connection_right, &m_resource_right, &m_stream_right, -1);
    if (r == CUDA_SUCCESS)
    {
        cuGraphicsResourceGetMappedEglFrame(&m_frame_right, m_resource_right, 0, 0);
    } else {
        printf ("Failed to acquire right image");
    }

}

ScopedCudaEGLStreamFrameAcquire::~ScopedCudaEGLStreamFrameAcquire()
{
    if (m_resource_left)
    {
        cuEGLStreamConsumerReleaseFrame(&m_connection_left, m_resource_left, &m_stream_left);
    }
    if (m_resource_right)
    {
        cuEGLStreamConsumerReleaseFrame(&m_connection_right, m_resource_right, &m_stream_right);
    }
}

bool ScopedCudaEGLStreamFrameAcquire::hasValidFrame() const
{
    return checkFormat();
}

bool ScopedCudaEGLStreamFrameAcquire::dumpImage(int frameNumber) {

    if (!hasValidFrame())
        ORIGINATE_ERROR("Invalid frame");
// Create surface from luminance channel.
    CUDA_RESOURCE_DESC cudaResourceDesc_left;
    memset(&cudaResourceDesc_left, 0, sizeof(cudaResourceDesc_left));
    cudaResourceDesc_left.resType = CU_RESOURCE_TYPE_ARRAY;
    cudaResourceDesc_left.res.array.hArray = m_frame_left.frame.pArray[0];
    CUsurfObject cudaSurfObj_left = 0;
    CUresult cuResult_left = cuSurfObjectCreate(&cudaSurfObj_left, &cudaResourceDesc_left);

    CUDA_RESOURCE_DESC cudaResourceDesc_right;
    memset(&cudaResourceDesc_right, 0, sizeof(cudaResourceDesc_right));
    cudaResourceDesc_right.resType = CU_RESOURCE_TYPE_ARRAY;
    cudaResourceDesc_right.res.array.hArray = m_frame_right.frame.pArray[0];
    CUsurfObject cudaSurfObj_right = 0;
    CUresult cuResult_right = cuSurfObjectCreate(&cudaSurfObj_right, &cudaResourceDesc_right);

    if (cuResult_left != CUDA_SUCCESS || cuResult_right != CUDA_SUCCESS)
    {
        ORIGINATE_ERROR("Unable to create the surface object (CUresult left: %s, right: %s)",
                         getCudaErrorString(cuResult_left), getCudaErrorString(cuResult_left));
    }
    // dumping images
    float time = 0.0f;
    auto start = std::chrono::high_resolution_clock::now();
    imageDumper(cuResult_left, cuResult_left, m_frame_left.width, m_frame_left.height, frameNumber);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken by imageDumper: "
            << duration.count() << " microseconds" << std::endl;

    // Destroy surface.
    cuSurfObjectDestroy(cudaSurfObj_left);
    cuSurfObjectDestroy(cudaSurfObj_right);
}


Size2D<uint32_t> ScopedCudaEGLStreamFrameAcquire::getSize() const
{
    if (hasValidFrame())
        return Size2D<uint32_t>(m_frame_left.width, m_frame_left.height);
    return Size2D<uint32_t>(0, 0);
}

bool ScopedCudaEGLStreamFrameAcquire::checkFormat() const
{
    if (!isCudaFormatYUV(m_frame_left.eglColorFormat))
    {
        ORIGINATE_ERROR("Only YUV color formats are supported");
    }
    if (m_frame_left.cuFormat != CU_AD_FORMAT_UNSIGNED_INT8)
    {
        ORIGINATE_ERROR("Only 8-bit unsigned int formats are supported");
    }

    if (!isCudaFormatYUV(m_frame_right.eglColorFormat))
    {
        ORIGINATE_ERROR("Only YUV color formats are supported");
    }
    if (m_frame_right.cuFormat != CU_AD_FORMAT_UNSIGNED_INT8)
    {
        ORIGINATE_ERROR("Only 8-bit unsigned int formats are supported");
    }
    return true;
}

static bool execute(const CommonOptions& options)
{
    // Initialize EGL.
    PROPAGATE_ERROR(g_display.initialize());

    // Initialize the Argus camera provider.
    UniqueObj<CameraProvider> cameraProvider(CameraProvider::create());

    // Get the ICameraProvider interface from the global CameraProvider.
    ICameraProvider *iCameraProvider = interface_cast<ICameraProvider>(cameraProvider);
    if (!iCameraProvider)
        ORIGINATE_ERROR("Failed to get ICameraProvider interface");
    printf("Argus Version: %s\n", iCameraProvider->getVersion().c_str());

    // Get the camera devices.
    std::vector<CameraDevice*> cameraDevices;
    iCameraProvider->getCameraDevices(&cameraDevices);
    if (cameraDevices.size() < 2)
        ORIGINATE_ERROR("Must have at least 2 sensors available");

    std::vector <CameraDevice*> lrCameras;
    lrCameras.push_back(cameraDevices[0]); // Left Camera (the 1st camera will be used for AC)
    lrCameras.push_back(cameraDevices[1]); // Right Camera

    // Create the capture session, AutoControl will be based on what the 1st device sees.
    UniqueObj<CaptureSession> captureSession(iCameraProvider->createCaptureSession(lrCameras));
    ICaptureSession *iCaptureSession = interface_cast<ICaptureSession>(captureSession);
    if (!iCaptureSession)
        ORIGINATE_ERROR("Failed to get capture session interface");

    // Create stream settings object and set settings common to both streams.
    UniqueObj<OutputStreamSettings> streamSettings(
        iCaptureSession->createOutputStreamSettings(STREAM_TYPE_EGL));
    IOutputStreamSettings *iStreamSettings = interface_cast<IOutputStreamSettings>(streamSettings);
    IEGLOutputStreamSettings *iEGLStreamSettings =
        interface_cast<IEGLOutputStreamSettings>(streamSettings);
    if (!iStreamSettings || !iEGLStreamSettings)
        ORIGINATE_ERROR("Failed to create OutputStreamSettings");
    iEGLStreamSettings->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
    iEGLStreamSettings->setResolution(STREAM_SIZE);
    iEGLStreamSettings->setEGLDisplay(g_display.get());

    // Create egl streams
    PRODUCER_PRINT("Creating left stream.\n");
    iStreamSettings->setCameraDevice(lrCameras[0]);
    UniqueObj<OutputStream> streamLeft(iCaptureSession->createOutputStream(streamSettings.get()));
    IEGLOutputStream *iStreamLeft = interface_cast<IEGLOutputStream>(streamLeft);
    if (!iStreamLeft)
        ORIGINATE_ERROR("Failed to create left stream");

    PRODUCER_PRINT("Creating right stream.\n");
    iStreamSettings->setCameraDevice(lrCameras[1]);
    UniqueObj<OutputStream> streamRight(iCaptureSession->createOutputStream(streamSettings.get()));
    IEGLOutputStream *iStreamRight = interface_cast<IEGLOutputStream>(streamRight);
    if (!iStreamRight)
        ORIGINATE_ERROR("Failed to create right stream");

    PRODUCER_PRINT("Launching disparity checking consumer\n");
    StereoDisparityConsumerThread disparityConsumer(iStreamLeft, iStreamRight);
    PROPAGATE_ERROR(disparityConsumer.initialize());
    PROPAGATE_ERROR(disparityConsumer.waitRunning());

    // Create a request
    UniqueObj<Request> request(iCaptureSession->createRequest());
    IRequest *iRequest = interface_cast<IRequest>(request);
    if (!iRequest)
        ORIGINATE_ERROR("Failed to create Request");

    // Enable both streams in the request.
    iRequest->enableOutputStream(streamLeft.get());
    iRequest->enableOutputStream(streamRight.get());

    // Submit capture for the specified time.
    PRODUCER_PRINT("Starting repeat capture requests.\n");
    if (iCaptureSession->repeat(request.get()) != STATUS_OK)
        ORIGINATE_ERROR("Failed to start repeat capture request for preview");
    sleep(options.captureTime());

    // Stop the capture requests and wait until they are complete.
    iCaptureSession->stopRepeat();
    iCaptureSession->waitForIdle();

    // Disconnect Argus producer from the EGLStreams (and unblock consumer acquire).
    PRODUCER_PRINT("Captures complete, disconnecting producer.\n");
    iStreamLeft->disconnect();
    iStreamRight->disconnect();

    // Wait for the consumer thread to complete.
    PROPAGATE_ERROR(disparityConsumer.shutdown());

    // Shut down Argus.
    cameraProvider.reset();

    // Cleanup the EGL display
    PROPAGATE_ERROR(g_display.cleanup());

    PRODUCER_PRINT("Done -- exiting.\n");
    return true;
}

}; // namespace ArgusSamples

int main(int argc, char *argv[])
{
    ArgusSamples::CommonOptions options(basename(argv[0]),
                                        ArgusSamples::CommonOptions::Option_T_CaptureTime);
    if (!options.parse(argc, argv))
        return EXIT_FAILURE;
    if (options.requestedExit())
        return EXIT_SUCCESS;

    if (!ArgusSamples::execute(options))
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
