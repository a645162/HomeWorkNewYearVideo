# OpenCL Kernel Functions Source conversion to C header file

# Replace Keyword "@input_content@"
function(convert_cl_to_c_header cl_file hpp_template_file)
    file(READ "${cl_file}" input_content)

    # 检查首尾是否已经包含换行符
    string(FIND "${cl_file}" "\n" has_newline_start)
    string(FIND "${cl_file}" "\n" has_newline_end REVERSE)

    # 如果没有在首部添加换行符
    if (has_newline_start LESS 0)
        string(CONCAT input_content "\n" "${input_content}")
    endif ()

    # 如果没有在尾部添加换行符
    if (has_newline_end LESS 0)
        string(CONCAT input_content "${input_content}" "\n")
    endif ()

    string(REGEX REPLACE "\\.hpp$" ".h" h_output_file ${hpp_template_file})

    # 配置文件
    configure_file(
            "${hpp_template_file}"
            "${h_output_file}"
            @ONLY
    )
endfunction()

# Image Crop
convert_cl_to_c_header(
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/ImageCrop.cl"
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelCrop.hpp"
)

# Image Rotate
convert_cl_to_c_header(
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/ImageRotate.cl"
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelRotate.hpp"
)

# Image Merge
convert_cl_to_c_header(
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/ImageMerge.cl"
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelMerge.hpp"
)

# Image Channel Convert
convert_cl_to_c_header(
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/ImageChannel.cl"
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelChannel.hpp"
)

# Image Resize
convert_cl_to_c_header(
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/ImageResize.cl"
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelImageResize.hpp"
)

# Image Mask
convert_cl_to_c_header(
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/ImageMaskCircle.cl"
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelMaskImage.hpp"
)

# Image Convolution
convert_cl_to_c_header(
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/ImageConvolution.cl"
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelConvolution.hpp"
)

# Image Gaussian Convolution Kernel
convert_cl_to_c_header(
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/GaussianConvKernel.cl"
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelGaussianConvKernel.hpp"
)
