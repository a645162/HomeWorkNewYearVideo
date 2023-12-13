//
// Created by Haomin Kong on 2023/12/10.
//

#include "CalcByCenter.hpp"

Rect getRectByCenterPoint(Point center, int width, int height) {
    return {
            center.x - width / 2,
            center.y - height / 2,
            center.x + width / 2,
            center.y + height / 2
    };
}

Circle getCircleByCenterPoint(Point center, int radius) {
    return {
            .r=radius,
            .center=center
    };
}

Point getCenterByRect(Rect rect) {
    return {
            .x=(rect.x1 + rect.x2) / 2,
            .y=(rect.y1 + rect.y2) / 2
    };
}

Point getLeftTopByRect(Rect rect) {
    return {
            .x=rect.x1,
            .y=rect.y1
    };
}