//
// Created by konghaomin on 23-12-15.
//

#include "OpenCVVideo.h"

void WriteFrame(cv::VideoWriter writer, cv::Mat* frame) {
    if (writer.isOpened() == false) {
        std::cout << "Error: VideoWriter is not opened." << std::endl;
        return;
    }

    writer.write(*frame);
}

void WriteFrame(const cv::VideoWriter& writer, cv::Mat& frame) {
    WriteFrame(writer, &frame);
}
