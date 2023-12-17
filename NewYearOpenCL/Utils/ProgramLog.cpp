// Program Log functions
// Created by Haomin Kong on 23-12-17.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ProgramLog.h"

#include <iostream>

void output_frame_log(
        unsigned int chapter, unsigned int section,
        unsigned int index, unsigned int total_frame
) {
    std::cout << "Chapter " << chapter << " Section " << section;
    std::cout << " Frame " << index + 1 << "/" << total_frame << std::endl;
}