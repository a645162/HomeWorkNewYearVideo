# OpenCL Kernel Functions Source conversion to C header file

# Image Resize
file(READ
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/ImageResize.cl"
        CL_KERNEL_RESIZE_IMAGE
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelImageResize.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelImageResize.h
        @ONLY
)

# Image Mask
file(READ
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/ImageMaskCircle.cl"
        CL_KERNEL_MASK_CIRCLE
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelMaskImage.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelMaskImage.h
        @ONLY
)

# Image Convolution
file(READ
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/ImageConvolution.cl"
        CL_KERNEL_CONVOLUTION
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelConvolution.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelConvolution.h
        @ONLY
)

# Image Gaussian Convolution Kernel
file(READ
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/GaussianConvKernel.cl"
        CL_KERNEL_GAUSSIAN_CONV_KERNEL
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelGaussianConvKernel.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelGaussianConvKernel.h
        @ONLY
)
