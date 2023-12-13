//
// Created by 孔昊旻 on 2023/11/21.
//

#include "create_video_1.h"

int create_video_1() {
    // 设置视频编解码器
    cv::VideoWriter outputVideo;
    cv::Size frameSize(640, 480);
    int framesPerSecond = FPS;
    outputVideo.open(
            "output.avi",
            cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
            framesPerSecond, frameSize
    );

    // 检查视频是否成功打开
    if (!outputVideo.isOpened()) {
        std::cout << "无法打开输出视频文件！" << std::endl;
        return -1;
    }

    cv::Mat baseFrame(frameSize, CV_8UC4);

//    cv::Mat resizedImage;
//    cv::resize(image, resizedImage, cv::Size(newWidth, newHeight));

//    cv::Mat pic = cv::imread("base.png");
//    cv::resize(pic, baseFrame, cv::Size(640, 480));

    // 逐帧写入视频
    for (int i = 0; i < TOTAL_FRAME; i++) {
        // 帧
//        cv::Mat frame(frameSize, CV_8UC4);
        cv::Mat frame = baseFrame.clone();

        // 生成纯黑色背景
        cv::Mat background(frame.size(), CV_8UC4);

        // 计算中心点
        int center_width = frame.size().width / 2;
        int center_height = frame.size().height / 2;

        // 边长
        auto a = (int) (100 * (i / 100.0));

        // 确定坐标
        const auto
                x1 = center_width - a,
                y1 = center_height - a,
                x2 = center_width + a,
                y2 = center_height + a;


        // 创建一个与图像大小相同的透明图像
        cv::Mat layer_overlay(
                frame.size(), CV_8UC4,
                cv::Scalar(0, 0, 0, 0));

//        // overlay 层绘制图形
//        cv::rectangle(
//                layer_overlay,
//                cv::Point(x1, y1),
//                cv::Point(x2, y2),
//                cv::Scalar(255, 255, 255),
//                BORDER
//        );
//        cv::circle(
//                layer_overlay,
//                cv::Point(center_width, center_height),
//                a,
//                cv::Scalar(
//                        255 * (1.0 * i / TOTAL_FRAME),
//                        255 - 255 * (1.0 * i / TOTAL_FRAME),
//                        255
//                ),
//                BORDER
//        );
//
//        const auto fly_x_start = 50, fly_x_end = 500;
//        auto current_x = (int) ((double) (fly_x_end - fly_x_start) * (1.0 * i / TOTAL_FRAME)) + fly_x_start;
//        cv::circle(
//                layer_overlay,
//                cv::Point(current_x, center_height),
//                50,
//                cv::Scalar(0, 255, 0),
//                -1
//        );

        double p = 1.0 * i / TOTAL_FRAME;





        // 将绘制的矩形与原始图像进行混合
        const double alpha_start = 50.0;
        double alpha = 255 * (
                (1.0 * i / TOTAL_FRAME)
        );
        if (alpha < alpha_start) {
            alpha = alpha_start;
        }

        cv::addWeighted(
                layer_overlay,
                alpha / 255.0,
                background,
                1.0 - alpha / 255.0,
                0,
                frame
        );

        // 写入帧
        outputVideo.write(frame);

        cv::imshow("Frame", frame);
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // 释放VideoWriter对象
    outputVideo.release();

    // 关闭窗口
    cv::destroyAllWindows();

    return 0;
}
