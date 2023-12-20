#message(STATUS "Test For Kernel Function")

file(GLOB_RECURSE PROGRAM_UTILS "../../Utils/*.cpp")
file(GLOB_RECURSE PROGRAM_UTILS_HEADER "../../Utils/*.h")

file(GLOB_RECURSE OPENCL_INCLUDE "../../OpenCL/Include/*.cpp")
file(GLOB_RECURSE OPENCL_INCLUDE_HEADER "../../OpenCL/Include/*.h")
file(GLOB_RECURSE OPENCL_DEVICE "../../OpenCL/Devices/*.cpp")
file(GLOB_RECURSE OPENCL_DEVICE_HEADER "../../OpenCL/Devices/*.h")

file(GLOB_RECURSE OPENCL_IMAGE "../../OpenCL/Image/*.cpp")
file(GLOB_RECURSE OPENCL_IMAGE_HEADER "../../OpenCL/Image/*.h")

add_executable(
        OpenCLTestSourceBuild

        KernelBuildCheck.cpp

        ${PROGRAM_UTILS}
        ${PROGRAM_UTILS_HEADER}

        ${OPENCL_INCLUDE}
        ${OPENCL_INCLUDE_HEADER}
        ${OPENCL_DEVICE}
        ${OPENCL_DEVICE_HEADER}

        ${OPENCL_IMAGE}
        ${OPENCL_IMAGE_HEADER}
)

# OpenCL
# Set OpenCL Version to 1.2
# Because AMD GPU and Apple Mac Series Devices is not support OpenCL 2.0 or higher.
target_compile_definitions(
        OpenCLTestSourceBuild PRIVATE
        CL_TARGET_OPENCL_VERSION=120
)
target_compile_definitions(
        OpenCLTestSourceBuild PRIVATE
        CL_MINIMUM_OPENCL_VERSION=120
)
target_compile_definitions(
        OpenCLTestSourceBuild PRIVATE
        CL_HPP_TARGET_OPENCL_VERSION=120
)
target_compile_definitions(
        OpenCLTestSourceBuild PRIVATE
        CL_HPP_MINIMUM_OPENCL_VERSION=120
)

target_link_libraries(
        OpenCLTestSourceBuild PRIVATE ${OpenCL_LIBRARIES}
)

add_test(NAME OpenCLTestSourceBuildTest COMMAND OpenCLTestSourceBuild)
