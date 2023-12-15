//
// Created by konghaomin on 23-12-15.
//

#include "OpenCVVideo.h"

cv::Mat lastFrame;

void WriteFrame(cv::VideoWriter writer, cv::Mat &frame) {
    writer.write(frame);
    lastFrame = frame.clone();
}