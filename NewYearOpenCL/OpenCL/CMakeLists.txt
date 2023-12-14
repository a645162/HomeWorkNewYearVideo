# OpenCL Kernel Functions Source conversion to C header file

function(convert_cl_to_c_header input_file output_file)
    file(READ "${input_file}" input_content)

    # 检查首尾是否已经包含换行符
    string(FIND "${input_content}" "\n" has_newline_start)
    string(FIND "${input_content}" "\n" has_newline_end REVERSE)

    # 如果没有在首部添加换行符
    if (has_newline_start LESS 0)
        string(CONCAT input_content "\n" "${input_content}")
    endif()

    # 如果没有在尾部添加换行符
    if (has_newline_end LESS 0)
        string(CONCAT input_content "${input_content}" "\n")
    endif()

    # 配置文件
    configure_file("${input_file}" "${output_file}" @ONLY)
endfunction()

#convert_cl_to_c_header(
#        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/ImageConvolution.cl"
#        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelConvolution.h"
#)

# Image Resize
file(READ
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/ImageResize.cl"
        CL_KERNEL_RESIZE_IMAGE
)
string(CONCAT CL_KERNEL_RESIZE_IMAGE "\n" "${CL_KERNEL_RESIZE_IMAGE}" "\n")
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
string(CONCAT CL_KERNEL_MASK_CIRCLE "\n" "${CL_KERNEL_MASK_CIRCLE}" "\n")
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
string(CONCAT CL_KERNEL_CONVOLUTION "\n" "${CL_KERNEL_CONVOLUTION}" "\n")
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
string(CONCAT CL_KERNEL_GAUSSIAN_CONV_KERNEL "\n" "${CL_KERNEL_GAUSSIAN_CONV_KERNEL}" "\n")
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelGaussianConvKernel.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelGaussianConvKernel.h
        @ONLY
)
