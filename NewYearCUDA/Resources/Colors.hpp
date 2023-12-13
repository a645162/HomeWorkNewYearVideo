//
// Created by Haomin Kong on 2023/12/10.
//

#include <opencv2/opencv.hpp>

typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} RGB;

typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
} RGBA;

// Shanghai Maritime University Color
// https://scm.shmtu.edu.cn/photo/
// CMYK 100 80 0 37
// RGB #0020A1 0 32 161
#define BGR_COLOR_SHMTU_BLUE 161, 32, 0
#define SHMTU_BLUE_RGB ((RGB)(.r=0, .g=32, .b=161))
#define SHMTU_BLUE_RGBA ((RGBA)(.r=0, .g=32, .b=161,.a=255))

uchar3 getColor(RGB color){
    return make_uchar3(color.r, color.g, color.b);
};