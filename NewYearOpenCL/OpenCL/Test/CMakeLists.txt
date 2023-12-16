#find_package(Eigen3 REQUIRED)
#INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
#message(STATUS "Eigen3 Include Dirs: ${EIGEN3_INCLUDE_DIR}")

add_executable(
        Test

        #        OpenCL/Test/progress_test.cpp
        #        OpenCL/Test/gaussian_eigen3.cpp
        #        OpenCL/Test/TestOpenCLMatrixAddFloat.cpp
        #        OpenCL/Test/TestOpenCLMatrixAddInt.cpp
        #        OpenCL/Test/TestOpenCLRect.cpp
)

# Eigen3
#target_link_libraries(Test PRIVATE Eigen3::Eigen)