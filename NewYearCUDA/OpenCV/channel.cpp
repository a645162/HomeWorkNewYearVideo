//
// Created by konghaomin on 2023/12/9.
//

#include "channel.h"

#include <opencv2/opencv.hpp>

cv::Mat convertTo4Channels(const cv::Mat& inputImage) {
    if (inputImage.channels() == 4) {
        return inputImage.clone();
    }

    cv::Mat outputImage;

    if (inputImage.channels() == 1) {
        cv::cvtColor(inputImage, outputImage, cv::COLOR_GRAY2BGRA);
    } else if (inputImage.channels() == 3) {
        cv::cvtColor(inputImage, outputImage, cv::COLOR_BGR2BGRA);
    } else {
        std::cerr << "Unsupported number of channels: " << inputImage.channels() << std::endl;
        return cv::Mat();
    }

    return outputImage;
}