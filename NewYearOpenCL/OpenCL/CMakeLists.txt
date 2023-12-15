# OpenCL Kernel Functions Source conversion to C header file

# Replace Keyword "@input_content@"
function(convert_cl_to_c_header cl_file hpp_template_file)
    file(READ "${cl_file}" input_content)

    # Check if file has newline at start or end.
    string(FIND "${cl_file}" "\n" has_newline_start)
    string(FIND "${cl_file}" "\n" has_newline_end REVERSE)

    # Add \t to start of file.
    if (has_newline_start LESS 0)
        string(CONCAT input_content "\n" "${input_content}")
    endif ()

    # Add \n to end of file.
    if (has_newline_end LESS 0)
        string(CONCAT input_content "${input_content}" "\n")
    endif ()

    # Convert .hpp to .h
    string(REGEX REPLACE "\\.hpp$" ".h" h_output_file ${hpp_template_file})

    # Replace keyword
    configure_file(
            "${hpp_template_file}"
            "${h_output_file}"
            @ONLY
    )
endfunction()

# Gradient Image
convert_cl_to_c_header(
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/GenerateGradientColor.cl"
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelGenerateGradientColor.hpp"
)
convert_cl_to_c_header(
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/GenerateGradientImage.cl"
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelGenerateGradientImage.hpp"
)

# Image Mirror
convert_cl_to_c_header(
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/ImageMirror.cl"
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelImageMirror.hpp"
)

# Image Gray RGB
convert_cl_to_c_header(
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/ImageGrayRGB.cl"
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelImageGrayRGB.hpp"
)

# Image Crop
convert_cl_to_c_header(
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/ImageCrop.cl"
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelImageCrop.hpp"
)

# Image Rotate
convert_cl_to_c_header(
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/ImageRotate.cl"
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelImageRotate.hpp"
)

# Image Merge
convert_cl_to_c_header(
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/ImageMerge.cl"
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelImageMerge.hpp"
)

# Image Channel Convert
convert_cl_to_c_header(
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/ImageChannel.cl"
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelImageChannel.hpp"
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
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelImageConvolution.hpp"
)

# Image Gaussian Convolution Kernel
convert_cl_to_c_header(
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/Source/GaussianConvKernel.cl"
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/KernelGaussianConvKernel.hpp"
)
