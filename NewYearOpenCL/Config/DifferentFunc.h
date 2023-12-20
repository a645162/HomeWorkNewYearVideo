// Different Function
// Created by Haomin Kong on 23-12-20.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef DIFFERENT_FUNC_H
#define DIFFERENT_FUNC_H

#ifdef MSVC_COMPILER

// MSVC
#define STRDUP_UNIVERSAL _strdup

#else

// MinGW or Other
#define STRDUP_UNIVERSAL strdup

#endif

#endif //DIFFERENT_FUNC_H
