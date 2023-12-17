// Program IO functions
// Created by Haomin Kong on 23-12-14.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ProgramIO.h"

#include <iostream>


void print_multi_char(const char chr, unsigned int length) {
    for (unsigned int i = 0; i < length; ++i) {
        std::cout << chr;
    }
    std::cout << std::endl;
}

void WaitForEnterPress() {
    std::cout << "Press 'Enter' to continue..." << std::endl;
    std::cin.get();
}