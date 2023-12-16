file(GLOB PROGRAM_IO "../../Utils/*.cpp")

file(GLOB OPENCL_INCLUDE "../Include/*.cpp")
file(GLOB OPENCL_DEVICE "../Devices/*.cpp")

file(GLOB OPENCL_IMAGE_OPERATION "../Image/*.cpp")
file(GLOB OPENCL_IMAGE_DRAW "../Image/Draw/*.cpp")
file(GLOB OPENCL_IMAGE_GENERATE "../Image/Generate/*.cpp")

file(GLOB DEMO_IMAGE_CONVOLUTION "ImageConvolution/*.cpp")

add_executable(
        Demo

        Demo.cpp

        # Some IO Operate
        ${PROGRAM_IO}

        # OpenCL Function
        ${OPENCL_INCLUDE}

        # OpenCL Device
        ${OPENCL_DEVICE}

        # OpenCL Image
        ${OPENCL_IMAGE_OPERATION}
        ${OPENCL_IMAGE_DRAW}
        ${OPENCL_IMAGE_GENERATE}

        # Image Convolution
        ${DEMO_IMAGE_CONVOLUTION}
)

# OpenCL
# Set OpenCL Version to 1.2
# Because AMD GPU and Apple Mac Series Devices is not support OpenCL 2.0 or higher.
target_compile_definitions(Demo PRIVATE CL_TARGET_OPENCL_VERSION=120)
target_compile_definitions(Demo PRIVATE CL_MINIMUM_OPENCL_VERSION=120)
target_compile_definitions(Demo PRIVATE CL_HPP_TARGET_OPENCL_VERSION=120)
target_compile_definitions(Demo PRIVATE CL_HPP_MINIMUM_OPENCL_VERSION=120)

target_link_libraries(Demo PRIVATE ${OpenCL_LIBRARIES})

# OpenCV
target_link_libraries(Demo PRIVATE ${OpenCV_LIBS})