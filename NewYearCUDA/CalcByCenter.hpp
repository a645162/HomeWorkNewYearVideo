//
// Created by Haomin Kong on 2023/12/10.
//

#ifndef NEW_YEAR_CUDA_CALC_BY_CENTER_H
#define NEW_YEAR_CUDA_CALC_BY_CENTER_H

typedef struct {
    int x;
    int y;
} Point;

typedef struct {
    int r;
    Point center;
} Circle;

typedef struct {
    int x1;
    int y1;
    int x2;
    int y2;
} Rect;

#endif //NEW_YEAR_CUDA_CALC_BY_CENTER_H
