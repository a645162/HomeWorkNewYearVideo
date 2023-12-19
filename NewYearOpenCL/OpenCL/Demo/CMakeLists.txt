file(GLOB_RECURSE PROGRAM_IO "../../Utils/*.cpp")
file(GLOB_RECURSE PROGRAM_IO_HEADER "../../Utils/*.cpp")

file(GLOB_RECURSE OPENCL_INCLUDE "../Include/*.cpp")
file(GLOB_RECURSE OPENCL_INCLUDE_HEADER "../Include/*.cpp")
file(GLOB_RECURSE OPENCL_DEVICE "../Devices/*.cpp")
file(GLOB_RECURSE OPENCL_DEVICE_HEADER "../Devices/*.cpp")

file(GLOB_RECURSE OPENCL_IMAGE_OPERATION "../Image/*.cpp")
file(GLOB_RECURSE OPENCL_IMAGE_OPERATION_HEADER "../Image/*.cpp")

file(GLOB_RECURSE DEMO_IMAGE "Image/*.cpp")
file(GLOB_RECURSE DEMO_IMAGE_HEADER "Image/*.cpp")
file(GLOB_RECURSE DEMO_IMAGE_DRAW "Draw/*.cpp")
file(GLOB_RECURSE DEMO_IMAGE_DRAW_HEADER "Draw/*.cpp")
file(GLOB_RECURSE DEMO_IMAGE_GENERATE "Generate/*.cpp")
file(GLOB_RECURSE DEMO_IMAGE_GENERATE_HEADER "Generate/*.cpp")

add_executable(
        OpenCL_Image_Demo

        DemoMain.cpp

        # Some IO Operate
        ${PROGRAM_IO}
        ${PROGRAM_IO_HEADER}

        # OpenCL Function
        ${OPENCL_INCLUDE}
        ${OPENCL_INCLUDE_HEADER}

        # OpenCL Device
        ${OPENCL_DEVICE}
        ${OPENCL_DEVICE_HEADER}

        # OpenCL Image
        ${OPENCL_IMAGE_OPERATION}
        ${OPENCL_IMAGE_OPERATION_HEADER}

        # Image Demo
        ${DEMO_IMAGE}
        ${DEMO_IMAGE_HEADER}
        ${DEMO_IMAGE_DRAW}
        ${DEMO_IMAGE_DRAW_HEADER}
        ${DEMO_IMAGE_GENERATE}
        ${DEMO_IMAGE_GENERATE_HEADER}
)

# OpenCL
# Set OpenCL Version to 1.2
# Because AMD GPU and Apple Mac Series Devices is not support OpenCL 2.0 or higher.
target_compile_definitions(OpenCL_Image_Demo PRIVATE CL_TARGET_OPENCL_VERSION=120)
target_compile_definitions(OpenCL_Image_Demo PRIVATE CL_MINIMUM_OPENCL_VERSION=120)
target_compile_definitions(OpenCL_Image_Demo PRIVATE CL_HPP_TARGET_OPENCL_VERSION=120)
target_compile_definitions(OpenCL_Image_Demo PRIVATE CL_HPP_MINIMUM_OPENCL_VERSION=120)

target_link_libraries(OpenCL_Image_Demo PRIVATE ${OpenCL_LIBRARIES})

# OpenCV
target_link_libraries(OpenCL_Image_Demo PRIVATE ${OpenCV_LIBS})