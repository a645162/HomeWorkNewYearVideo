cmake_minimum_required(VERSION 3.22)

# Add Command Line Option to CMake on Ubuntu
# -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

project(NewYearCUDA CXX CUDA)
set(CMAKE_CUDA_STANDARD 20)

if (WIN32)
    set(CMAKE_TOOLCHAIN_FILE D:/Dev/vcpkg/scripts/buildsystems/vcpkg.cmake)
elseif (APPLE)
    set(CMAKE_TOOLCHAIN_FILE /Users/konghaomin/vcpkg/vcpkg/scripts/buildsystems/vcpkg.cmake)
endif ()

if (WIN32)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message(STATUS "Using MSVC compiler on Windows.")
        message(STATUS "${CMAKE_CXX_COMPILER_ID}")
        # https://blog.csdn.net/liangshui999/article/details/128332182
        message(STATUS "Set MSVC compiler to use UTF-8 as the character set.")
#        add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>")
        #add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/execution-charset:gbk>")
    else ()
        message(STATUS "Not using MSVC compiler on Windows.")
        message(STATUS "Program may not compatible with MinGW or Other compilers on Windows.")
    endif ()
endif ()

if (DEFINED CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg.cmake")
    message(STATUS "Using vcpkg toolchain.")
    message(STATUS "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
    set(USING_VCPKG ON)
else ()
    message(STATUS "Not using vcpkg.")
    set(USING_VCPKG OFF)
endif ()

########################################################################
# Fast Math on Floating Point Calculation
set(FAST_MATH ON)
set(FAST_MATH_CUDA OFF)
#################################
if (FAST_MATH)
    if (WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast")
    else ()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
    endif ()
endif ()

if (FAST_MATH_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
endif ()
########################################################################


########################################################################
# OpenCV Static Library
#################################
# Here should be same as OpenCV build setting.
set(BUILD_SHARED_LIBS ON)

if (NOT DEFINED OpenCV_STATIC)
    # look for global setting
    if (NOT DEFINED BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
        set(OpenCV_STATIC OFF)
    else ()
        set(OpenCV_STATIC ON)
    endif ()
endif ()
########################################################################

if (Win32 OR CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "Configuring for Windows")

    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        # Release Channel use CUDA 12.0 to get most compatibility.
        set(CMAKE_CUDA_COMPILER D:/Dev/GPU/CUDA/v12.0/bin/nvcc.exe)
    else ()
        # Debug
        #        set(CMAKE_CUDA_COMPILER D:/Dev/GPU/CUDA/v11.8/bin/nvcc.exe)
        set(CMAKE_CUDA_COMPILER D:/Dev/GPU/CUDA/V12.2/bin/nvcc.exe)
    endif ()

    if (NOT USING_VCPKG)
        #        set(OpenCV_DIR D:/Dev/Lib/OpenCV/opencv_contrib_cuda_4.8.0.20230804_win_amd64)
        #        set(OpenCV_DIR D:/Dev/Lib/OpenCV/OpenCV4.8.1WinindowsEXE/opencv/build)
        set(OpenCV_DIR D:/Dev/Lib/OpenCV/Official/OpenCV4.5.4/opencv/build)
        #        set(OpenCV_DIR D:/Dev/Lib/OpenCV/git/build-win64/install)
        #        set(OpenCV_DIR D:/Dev/Lib/OpenCV/opencv-4.8.1-WindowsBuild/build)
        message(STATUS "User OpenCV_DIR: ${OpenCV_DIR}")
    else ()
        message(STATUS "Using vcpkg to find OpenCV.")
    endif ()
elseif (CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(STATUS "Configuring for Linux")
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
elseif (APPLE OR CMAKE_SYSTEM_NAME MATCHES "Darwin")
    message(FATAL_ERROR "Program not support Apple Platform ${CMAKE_SYSTEM_NAME}")
    message(FATAL_ERROR "Program may not work correctly.")
    message(FATAL_ERROR "Apple Platform may not support CUDA.")
    return()
else ()
    message(FATAL_ERROR "Unknown Platform ${CMAKE_SYSTEM_NAME}")
    message(FATAL_ERROR "Program may not work correctly.")
    return()
endif ()

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    message(STATUS "Found CUDA!")

    # 重新获取CUDA编译器版本信息
    #    execute_process(COMMAND ${CMAKE_CUDA_COMPILER} --version OUTPUT_VARIABLE CUDA_COMPILER_VERSION_OUTPUT)
    #    message(STATUS "CUDA_COMPILER_VERSION_OUTPUT:\n ${CUDA_COMPILER_VERSION_OUTPUT}")
    #    string(REGEX MATCH "[0-9]{2}\.[0-9]" CUDA_COMPILER_VERSION ${CUDA_COMPILER_VERSION_OUTPUT})

    message(STATUS "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "CMAKE_CUDA_STANDARD: ${CMAKE_CUDA_STANDARD}")

    # Set Target Architecture for different CUDA version.
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "10.0")
        message(FATAL_ERROR "CUDA version must be greater than 10.0")
    elseif (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.0")
        message(FATAL_ERROR "CUDA version must be greater than 11.0")
    elseif (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.0")
        set(CMAKE_CUDA_ARCHITECTURES 50;52;60;61;70;75;80;86;)
    elseif (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.0")
        set(CMAKE_CUDA_ARCHITECTURES 50;52;60;61;70;75;80;86;89;90)
    else ()
        message(STATUS "CUDA version greater than 12")
        message(STATUS "But this program was built with CUDA 12.0 on 2023.")
        message(STATUS "So, this program arch support is limited.")
        message(STATUS "You should modify CMakeLists.txt to add new arch support.")
        message(STATUS "The latest support Game GPU is RTX 40 Series.")
        set(CMAKE_CUDA_ARCHITECTURES 50;52;60;61;70;75;80;86;89;90)
    endif ()

    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
    if (DEFINED CMAKE_CUDA_ARCHITECTURES)
        message(STATUS "CUDA ARCHITECTURES Support:")
        message(STATUS "\tGeForce:")
        if (CMAKE_CUDA_ARCHITECTURES MATCHES "89")
            message(STATUS "\t\tGeForce RTX 40 Series")
        endif ()
        if (CMAKE_CUDA_ARCHITECTURES MATCHES "80")
            message(STATUS "\t\tGeForce RTX 30 Series")
        endif ()
        if (CMAKE_CUDA_ARCHITECTURES MATCHES "75")
            message(STATUS "\t\tGeForce RTX 20 Series")
            message(STATUS "\t\tGeForce GTX 16 Series")
        endif ()
        if (CMAKE_CUDA_ARCHITECTURES MATCHES "61")
            message(STATUS "\t\tGeForce GTX 10 Series")
        endif ()
        if (CMAKE_CUDA_ARCHITECTURES MATCHES "52")
            message(STATUS "\t\tGeForce GTX 9 Series")
        endif ()
        if (CMAKE_CUDA_ARCHITECTURES MATCHES "50")
            message(STATUS "\t\tGeForce GTX 8 Series(NoteBook)")
            message(STATUS "\t\tGeForce GTX 750/750 Ti")
        endif ()
    endif ()
    message(STATUS "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
else ()
    message(FATAL_ERROR "Not found CUDA!")
endif ()

# Close warning because of OpenCV (build with CUDA) old CMake write policy.
#cmake_policy(SET CMP0146 OLD)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("OpenCV version: ${OpenCV_VERSION}")

add_executable(NewYearCUDA
        cuda_include.cuh
        #        opencv_include.cuh

#        main.cu
#        gaussian_blur_video.cu
#        channel.cu
        gray.cu
        #        matrix_add.cu

        #        focus.cu
)

set_target_properties(NewYearCUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

#set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "35;50")

target_link_libraries(NewYearCUDA ${OpenCV_LIBS})