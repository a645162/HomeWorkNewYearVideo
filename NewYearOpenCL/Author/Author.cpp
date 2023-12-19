// Author.cpp
// Show Author Information
// Created by Haomin Kong on 2023-12-13.
// https://github.com/a645162/HomeWorkNewYearVideo
// CopyRight (c) 2023 Shanghai Maritime University

#include "Author.h"
#include "../Utils/ProgramIO.h"

#include <iostream>

void KHM::sayHello() {
    std::cout << "========================================================\n";
    std::cout << "    Shanghai Maritime University\n";
    std::cout << "    New Year Card\n";
    std::cout << "    Author: Haomin Kong\n";
    std::cout << "    https://github.com/a645162/HomeWorkNewYearVideo\n";
    std::cout << "========================================================\n";
    WaitForEnterPress();
}
