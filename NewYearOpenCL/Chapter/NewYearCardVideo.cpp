// Project: New Year Card Video
// Created by Haomin Kong on 23-12-17.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "NewYearCardVideo.h"
#include "../Utils/ProgramIO.h"
#include "Chapter_1/Chapter.1.h"
#include "../Config/Config.h"

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <ctime>
#include <chrono>

float RatioVideoScale = DEFAULT_RESOLUTION_SCALE_RATIO;
float RatioVideoFrame = DEFAULT_FRAME_RATE_SCALE_RATIO;

int CANVAS_WIDTH = ORIGIN_CANVAS_WIDTH, CANVAS_HEIGHT = ORIGIN_CANVAS_HEIGHT;
int CANVAS_CENTER_X = ORIGIN_CANVAS_WIDTH / 2, CANVAS_CENTER_Y = ORIGIN_CANVAS_HEIGHT / 2;
int FRAME_RATE = DEFAULT_FRAME_RATE;

void start_generate(cl_device_id device, cl_context context) {
    cv::VideoWriter outputVideo;
    cv::Size frameSize(CANVAS_WIDTH, CANVAS_HEIGHT);

    char file_name[1024] = "output.avi";

    std::time_t currentTime = std::time(nullptr);
    char timeBuffer[80];
    const char *format = "%Y_%m_%d_%H_%M_%S";
    std::strftime(timeBuffer, sizeof(timeBuffer), format, std::localtime(&currentTime));
    if (std::strftime(timeBuffer, sizeof(timeBuffer), format, std::localtime(&currentTime))) {
        std::cout << "FileTime:" << timeBuffer << std::endl;
        snprintf(file_name, sizeof(file_name), "NewYearCardVideo_%s.avi", timeBuffer);
    }

    std::cout << "File Name: " << file_name << std::endl;

    outputVideo.open(
            file_name,
            cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
            FRAME_RATE, frameSize
    );

    if (outputVideo.isOpened()) {
        std::cout << "Open file success!" << std::endl;
    } else {
        std::cout << "Cannot open file!" << std::endl;
        exit(1);
    }

    // Chapter 1
    chapter_1(context, device, 600 * RatioVideoFrame, outputVideo);

    outputVideo.release();

    cv::destroyAllWindows();
}

void video_main(cl_device_id device, cl_context context) {
    std::cout << "New Year Card Video" << std::endl;

    // Main Program
#ifndef DEBUG_MODE
    RatioVideoScale = UserInputWithDefault("Video Scale Ratio", DEFAULT_RESOLUTION_SCALE_RATIO);
    RatioVideoFrame = UserInputWithDefault("Video Frame Ratio", DEFAULT_FRAME_RATE_SCALE_RATIO);
    FRAME_RATE = UserInputWithDefault("Video Frame Rate", DEFAULT_FRAME_RATE);
#endif

    CANVAS_WIDTH = (int) (ORIGIN_CANVAS_WIDTH * RatioVideoScale);
    CANVAS_HEIGHT = (int) (ORIGIN_CANVAS_HEIGHT * RatioVideoScale);

    CANVAS_CENTER_X = CANVAS_WIDTH / 2;
    CANVAS_CENTER_Y = CANVAS_HEIGHT / 2;

    FRAME_RATE = (int) (DEFAULT_FRAME_RATE * RatioVideoFrame);

    std::cout << "Ready to generate video" << std::endl;
#ifndef DEBUG_MODE
    WaitForEnterPress();
#endif

    auto startTime = std::chrono::high_resolution_clock::now();

    start_generate(device, context);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    double seconds = duration.count() / 1000.0;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Run Time: " << seconds << " S" << std::endl;

    std::cout << "New Year Card Video End" << std::endl;
}