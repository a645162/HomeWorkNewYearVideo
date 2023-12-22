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

project(NewYear_OpenCL C CXX)
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

# Compiler Options
# Add character set option for MSVC on Windows to support Chinese.
if (WIN32)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message(STATUS "@Using MSVC compiler on Windows.")
        # https://blog.csdn.net/liangshui999/article/details/128332182
        message(STATUS "Set MSVC compiler to use UTF-8 as the character set.")
        add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>")
        #add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/execution-charset:gbk>")

        # Disable Warning C4996
        add_definitions(-D_CRT_SECURE_NO_WARNINGS)

        add_definitions(-DMSVC_COMPILER)
    else ()
        message(STATUS "Not using MSVC compiler on Windows.")
        message(STATUS "!!!Program may not compatible with MinGW or Other compilers on Windows.")
    endif ()
endif ()

if (WIN32)
    set(OpenCV_DIR "../../OpenCV-4.8.0-official")
    #    set(OpenCV_DIR "../../OpenCV-MinGW-Build-OpenCV-4.5.5-x64")
endif ()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "@OpenCV Version: ${OpenCV_VERSION}")

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
message(STATUS "OpenCL Include Dirs: ${OpenCL_INCLUDE_DIRS}")

# Use to generate C Header include OpenCL Kernel.
add_subdirectory(OpenCL)

file(GLOB_RECURSE CONFIG_HEADERS "Config/*.h")

file(GLOB_RECURSE PROGRAM_UTILS "Utils/*.cpp")

file(GLOB_RECURSE OPENCL_INCLUDE "OpenCL/Include/*.cpp")
file(GLOB_RECURSE OPENCL_INCLUDE_HEADER "OpenCL/Include/*.h")
file(GLOB_RECURSE OPENCL_DEVICE "OpenCL/Devices/*.cpp")
file(GLOB_RECURSE OPENCL_DEVICE_HEADER "OpenCL/Devices/*.h")

file(GLOB_RECURSE OpenCL_Utils "OpenCL/Utils/*.cpp")
file(GLOB_RECURSE OpenCL_Utils_HEADER "OpenCL/Utils/*.h")

file(GLOB_RECURSE OPENCL_IMAGE_OPERATION "OpenCL/Image/*.cpp")
file(GLOB_RECURSE OPENCL_IMAGE_OPERATION_HEADER "OpenCL/Image/*.h")

file(GLOB_RECURSE OPENCL_VIDEO_CHAPTER "Chapter/*.cpp")
file(GLOB_RECURSE OPENCL_VIDEO_CHAPTER_HEADER "Chapter/*.h")

file(GLOB_RECURSE ThirdParty "ThirdParty/*.cpp")
file(GLOB_RECURSE ThirdParty_HEADER "ThirdParty/*.h")

add_executable(
        ${PROJECT_NAME}

        main.cpp

        # Author Information
        Author/Author.cpp

        # Some Program Utils
        ${CONFIG_HEADERS}
        ${PROGRAM_UTILS}

        # OpenCV Function
        OpenCV/Video/OpenCVVideo.cpp

        # OpenCL Function
        ${OPENCL_INCLUDE}
        ${OPENCL_INCLUDE_HEADER}

        # OpenCL Device
        ${OPENCL_DEVICE}
        ${OPENCL_DEVICE_HEADER}

        ${OpenCL_Utils}
        ${OpenCL_Utils_HEADER}

        # OpenCL Image Process
        # + OpenCL Image Generate
        # + OpenCL Image Draw
        ${OPENCL_IMAGE_OPERATION}
        ${OPENCL_IMAGE_OPERATION_HEADER}

        # My Video Chapter
        ${OPENCL_VIDEO_CHAPTER}
        ${OPENCL_VIDEO_CHAPTER_HEADER}

        # Third Party
        ${ThirdParty}
        ${ThirdParty_HEADER}
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
target_link_libraries(${PROJECT_NAME} PRIVATE opencv_core opencv_highgui opencv_videoio)


# AVX2 SIMD Optimization
if (WIN32)
    if (MSVC)
        # Enable AVX2 for MSVC
        message(STATUS "Try to open AVX2 for MSVC on Windows")
        # Useful for MSVC On Windows
        target_compile_options(${PROJECT_NAME} PRIVATE "/arch:AVX2")
    elseif (CMAKE_COMPILER_IS_GNUCXX)
        message(STATUS "Try to open AVX2 for GCC on Windows")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx2 -mfma")
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(STATUS "Try to open AVX2 for Clang on Windows")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
        #    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif ()
elseif (APPLE)
    if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")
        message(STATUS "Target architecture is Apple Silicon (arm64)")
    else ()
        message(STATUS "Target architecture is Intel x86_64")
        if (CMAKE_COMPILER_IS_GNUCXX)
            message(STATUS "Try to open AVX2 for GCC On Intel Chip macOS")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx2 -mfma")
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message(STATUS "Try to open AVX2 for Clang On Intel Chip macOS")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
            #    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
        endif ()
    endif ()
else ()
    if (CMAKE_COMPILER_IS_GNUCXX)
        message(STATUS "Try to open AVX2 for GCC on Linux")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx2 -mfma")
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(STATUS "Try to open AVX2 for Clang on Linux")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
        #    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif ()
endif ()

# Demo
add_subdirectory(OpenCL/Demo)

include(CTest)
add_subdirectory(CMake/OpenCL)

set(TOKEI_OUTPUT ON)

if (TOKEI_OUTPUT)

# Tokei
set(TOKEI_EXECUTABLE tokei)

execute_process(
        COMMAND ${TOKEI_EXECUTABLE} ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE TOKEI_RESULT
        OUTPUT_VARIABLE TOKEI_OUTPUT
)

# Step 3: Save output to a file
file(WRITE Tokei.txt "${TOKEI_OUTPUT}")

# Display a message indicating the result
if (TOKEI_RESULT EQUAL 0)
    message("Tokei execution successful. Output saved to Tokei.txt")
else ()
    message(FATAL_ERROR "Tokei execution failed with code ${TOKEI_RESULT}")
endif ()

endif ()

message(STATUS "@Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "@Compiler Path: ${CMAKE_CXX_COMPILER}")
message(STATUS "@Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "@System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")
message(STATUS "@CMake Generator: ${CMAKE_GENERATOR}")
message(STATUS "@CMake Build tool: ${CMAKE_BUILD_TOOL}")
message(STATUS "@CMake Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "@CMake CXX Flags: ${CMAKE_CXX_FLAGS}")