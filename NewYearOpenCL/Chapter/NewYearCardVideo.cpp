// Project: New Year Card Video
// Created by Haomin Kong on 23-12-17.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "NewYearCardVideo.h"
#include "../Utils/ProgramIO.h"
#include "../Config/Config.h"

#include "Chapter_1/Chapter.1.h"
#include "Chapter_2/Chapter.2.h"

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

#define ENABLE_CHAPTER_1
#define ENABLE_CHAPTER_2

int CalcFrame(const int frame_length) {
    return static_cast<int>(static_cast<float>(frame_length) * RatioVideoFrame);
}

void start_generate(cl_device_id device, cl_context context) {
    cv::VideoWriter outputVideo;
    const cv::Size frame_size(CANVAS_WIDTH, CANVAS_HEIGHT);

    char file_name[1024] = "output.avi";

    const std::time_t currentTime = std::time(nullptr);
    char timeBuffer[80];
    const auto format = "%Y_%m_%d_%H_%M_%S";
    std::strftime(timeBuffer, sizeof(timeBuffer), format, std::localtime(&currentTime));
    if (std::strftime(timeBuffer, sizeof(timeBuffer), format, std::localtime(&currentTime))) {
        std::cout << "File Time:" << timeBuffer << std::endl;
        snprintf(file_name, sizeof(file_name), "NewYearCardVideo_%s.avi", timeBuffer);
    }

    std::cout << "File Name: " << file_name << std::endl;

    outputVideo.open(
        file_name,
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
        FRAME_RATE, frame_size
    );

    if (outputVideo.isOpened()) {
        std::cout << "Open file success!" << std::endl;
    } else {
        std::cout << "Cannot open file!" << std::endl;
        exit(1);
    }

    cv::Mat last_frame;
    // Chapter 1
#ifdef ENABLE_CHAPTER_1
    last_frame = chapter_1(context, device, CalcFrame(800), &outputVideo);
#else
    // White Canvas
    last_frame = cv::Mat(CANVAS_HEIGHT, CANVAS_WIDTH, CV_8UC3, cv::Scalar(255, 255, 255));
#endif

#ifdef ENABLE_CHAPTER_2
    last_frame = chapter_2(context, device, CalcFrame(600), &outputVideo, &last_frame);
#endif

    outputVideo.release();

    cv::destroyAllWindows();
}

void video_main(cl_device_id device, cl_context context) {
    std::cout << "New Year Card Video" << std::endl;

    // Main Program
#ifndef DEBUG_MODE
    std::cout << "Please Set Ratio" << std::endl;

    // There is a error,but it will be work when I add this line.
    UserInputWithDefault("", DEFAULT_RESOLUTION_SCALE_RATIO);

    RatioVideoScale = UserInputWithDefault("Video Scale Ratio", DEFAULT_RESOLUTION_SCALE_RATIO);
    if (RatioVideoScale < 0.1) {
        RatioVideoScale = 0.1f;
        std::cout << "Video Scale Ratio is too small, set to 0.1" << std::endl;
    }

    RatioVideoFrame = UserInputWithDefault("Video Frame Ratio", DEFAULT_FRAME_RATE_SCALE_RATIO);
    if (RatioVideoFrame < 0.1) {
        RatioVideoFrame = 0.1f;
        std::cout << "Video Frame Ratio is too small, set to 0.1" << std::endl;
    }

    FRAME_RATE = UserInputWithDefault("Video Frame Rate", DEFAULT_FRAME_RATE);
    if (FRAME_RATE < 1) {
        FRAME_RATE = 10;
        std::cout << "Video Frame Rate is too small(small than 10), set to 10" << std::endl;
    }
#endif

    CANVAS_WIDTH = static_cast<int>(ORIGIN_CANVAS_WIDTH * RatioVideoScale);
    CANVAS_HEIGHT = static_cast<int>(ORIGIN_CANVAS_HEIGHT * RatioVideoScale);

    CANVAS_CENTER_X = CANVAS_WIDTH / 2;
    CANVAS_CENTER_Y = CANVAS_HEIGHT / 2;

    FRAME_RATE = static_cast<int>(DEFAULT_FRAME_RATE * RatioVideoFrame);

    std::cout << "Canvas Size: " << CANVAS_WIDTH << " x " << CANVAS_HEIGHT << std::endl;
    std::cout << "Frame Rate: " << FRAME_RATE << std::endl;
    std::cout << "Canvas Center: " << "(" << CANVAS_CENTER_X << "," << CANVAS_CENTER_Y << ")" << std::endl;

    std::cout << "Ready to generate video" << std::endl;
#ifndef DEBUG_MODE
    WaitForEnterPress();
#endif

    const auto startTime = std::chrono::high_resolution_clock::now();

    start_generate(device, context);

    const auto endTime = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    const double seconds = static_cast<double>(duration.count()) / 1000.0;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Run Time: " << seconds << " S" << std::endl;

    std::cout << "New Year Card Video End" << std::endl;
}
