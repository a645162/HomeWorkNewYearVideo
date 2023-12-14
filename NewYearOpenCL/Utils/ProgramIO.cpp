//
// Created by konghaomin on 23-12-14.
//

#include "ProgramIO.h"

#include <iostream>


void print_multi_char(const char chr, unsigned int length) {
    for (unsigned int i = 0; i < length; ++i) {
        std::cout << chr;
    }
    std::cout << std::endl;
}
