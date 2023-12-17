// Program IO functions
// Created by Haomin Kong on 23-12-14.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ProgramIO.h"

#include <iostream>
#include <sstream>


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

template <typename T>
T UserInputWithDefault(const char *prompt, T defaultValue) {
    std::string userInput;
    T value;

    std::cout << prompt << "(Default value: " << defaultValue << ")" << std::endl;
    std::getline(std::cin, userInput);

    if (userInput.empty()) {
        // User Input Empty, try to use default value
        value = defaultValue;
    } else {
        std::istringstream iss(userInput);
        if (iss >> value) {
            // User Input Corrects
        } else {
            // User Input Invalid
            value = defaultValue;
        }
    }

    return value;
}

template int UserInputWithDefault<int>(const char *prompt, int defaultValue);
template float UserInputWithDefault<float>(const char *prompt, float defaultValue);
template double UserInputWithDefault<double>(const char *prompt, double defaultValue);
