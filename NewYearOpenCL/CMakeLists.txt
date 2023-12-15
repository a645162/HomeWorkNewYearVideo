cmake_minimum_required(VERSION 3.12)
message(STATUS "@CMake version: ${CMAKE_VERSION}")

#if (WIN32)
#    message(STATUS "@Use VCPkg on Windows")
#
#    set(CMAKE_TOOLCHAIN_FILE "D:/Dev/vcpkg/scripts/buildsystems/vcpkg.cmake")
#    set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "")
#    if(DEFINED ENV{VCPKG_DEFAULT_TRIPLET} AND NOT DEFINED VCPKG_TARGET_TRIPLET)
#        set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_DEFAULT_TRIPLET}" CACHE STRING "")
#    endif()
#endif ()

project(NewYearOpenCL C CXX)
message(STATUS "@Project name: ${PROJECT_NAME}")

set(CMAKE_CXX_STANDARD 23)

message(STATUS "@Build type: ${CMAKE_BUILD_TYPE}")
#if (CMAKE_BUILD_TYPE STREQUAL "Debug")
if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Buildind in Debug Mode.")
    message(STATUS "\t#define DEBUG_MODE")
    add_definitions(-DDEBUG_MODE)
    #    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG_MODE)
endif ()

if (WIN32)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message(STATUS "@Using MSVC compiler on Windows.")

        # https://blog.csdn.net/liangshui999/article/details/128332182
        message(STATUS "Set MSVC compiler to use UTF-8 as the character set.")
        add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>")
        #add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/execution-charset:gbk>")
    else ()
        message(STATUS "Not using MSVC compiler on Windows.")
        message(STATUS "!!!Program may not compatible with MinGW or Other compilers on Windows.")
    endif ()
endif ()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "@OpenCV Version: ${OpenCV_VERSION}")

#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/kernel_template.h
#        ${CMAKE_CURRENT_BINARY_DIR}/kernel_template.hpp @ONLY)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
message(STATUS "OpenCL Include Dirs: ${OpenCL_INCLUDE_DIRS}")

# Use to generate C Header include OpenCL Kernel.
add_subdirectory(OpenCL)

add_executable(
        ${PROJECT_NAME}

        main.cpp

        Author/Author.cpp
        Author/Author.h

        Config/Config.h
        Config/Path.h
        Config/DebugVar.h

        Utils/ProgramIO.cpp
        Utils/ProgramIO.h

        OpenCL/Include/OpenCLInclude.cpp
        OpenCL/Include/OpenCLInclude.h
        OpenCL/Include/OpenCLError.cpp
        OpenCL/Include/OpenCLError.h
        OpenCL/Include/OpenCLFlow.cpp
        OpenCL/Include/OpenCLFlow.h
        OpenCL/Include/OpenCLProgram.cpp
        OpenCL/Include/OpenCLProgram.h

        OpenCL/Devices/OpenCLDevices.cpp
        OpenCL/Devices/OpenCLDevices.h
        OpenCL/Devices/OpenCLDevicesList.cpp
        OpenCL/Devices/OpenCLDevicesList.h

#        OpenCL/Image/ImageMask.cpp
#        OpenCL/Image/ImageMask.h
#        OpenCL/Kernel/KernelMaskImage.h

#        OpenCL/Image/ImageResize.cpp
#        OpenCL/Image/ImageResize.h
#        OpenCL/Kernel/KernelImageResize.h

#        OpenCL/Image/ImageConvolution.cpp
#        OpenCL/Image/ImageConvolution.h
#        OpenCL/Kernel/KernelConvolution.h

        OpenCL/Image/ImageCrop.cpp
        OpenCL/Image/ImageCrop.h
        OpenCL/Kernel/KernelCrop.h


        #        OpenCL/Image/ImageGaussianBlur.cpp
#        OpenCL/Image/ImageGaussianBlur.h
#        OpenCL/Kernel/KernelGaussianConvKernel.h

        #        OpenCL/Test/TestOpenCLRect.cpp
)

# OpenCL
# Set OpenCL Version to 1.2
# Because AMD GPU and Apple Mac Series Devices is not support OpenCL 2.0 or higher.
target_compile_definitions(${PROJECT_NAME} PRIVATE CL_TARGET_OPENCL_VERSION=120)
target_compile_definitions(${PROJECT_NAME} PRIVATE CL_MINIMUM_OPENCL_VERSION=120)
target_compile_definitions(${PROJECT_NAME} PRIVATE CL_HPP_TARGET_OPENCL_VERSION=120)
target_compile_definitions(${PROJECT_NAME} PRIVATE CL_HPP_MINIMUM_OPENCL_VERSION=120)

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCL_LIBRARIES})

# OpenCV
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
