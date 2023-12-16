add_executable(
        Demo

        Demo.cpp

        # Some IO Operate
        ../../Utils/ProgramIO.cpp
        ../../Utils/ProgramIO.h

        # OpenCL Function
        ../Include/OpenCLInclude.cpp
        ../Include/OpenCLInclude.h

        ../Include/OpenCLError.cpp
        ../Include/OpenCLError.h

        ../Include/OpenCLFlow.cpp
        ../Include/OpenCLFlow.h

        ../Include/OpenCLProgram.cpp
        ../Include/OpenCLProgram.h

        ../Include/OpenCLMem.cpp
        ../Include/OpenCLMem.h

        # OpenCL Device
        ../Devices/OpenCLDevices.cpp
        ../Devices/OpenCLDevices.h

        ../Devices/OpenCLDevicesList.cpp
        ../Devices/OpenCLDevicesList.h

        # Image Convolution

        ../Image/ImageConvolution.cpp

        ImageConvolution/ImageConvolutionDemo.cpp
        ImageConvolution/ImageConvolutionDemo.h

        ImageConvolution/ImageGaussianBlurDemo.cpp
        ImageConvolution/ImageGaussianBlurDemo.h
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