# OpenCL Platform

Last Updated: 2023-12-12
Author: Haomin Kong

## Tested Platforms

### Summary

Include the following types of platforms:

- Intel Core and Xeon CPU
- Intel Integrated Graphics
- AMD Radeon GPU
- NVIDIA GeForce and Tesla GPU

Include the following platforms:

- Intel(R) Core(TM) i7-8850H CPU @ 2.60GHz(Very Slow)
- Intel UHD Graphics 630 1536 MB
- AMD Radeon Pro RX 560X 4 GB
- AMD Radeon Vega 64 8 GB(gfx900)
- Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
- NVIDIA Tesla P40 24 GB
- NVIDIA GTX 1660 Super 6 GB
- NVIDIA P102 10 GB

### My PC

- Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
- 64 GB RAM 2400 MHz Reg ECC DDR4(4 x 16GB works in quad-channel mode)
- AMD Radeon Vega 64 8 GB(gfx900)
- NVIDIA Tesla P40 24 GB
- NVIDIA GTX 1660 Super 6 GB
- NVIDIA P102 10 GB

#### Windows 11 Enterprise 23H2

- Microsoft Visual Studio 2022 17.8.3
- CMake 3.27.0(Bundled with CLion 2023.3.1)
- OpenCV 4.8.0
- CUDA 12.3.1

#### Ubuntu 22.04.3 LTS

- Linux Kernel 6.2.0-39-generic
- CMake 3.22.1
- OpenCV 4.5.4
- CUDA 12.3.1

Notice: Most packages are installed from apt repository.

### My MacBook Pro 15' 2018

- Intel(R) Core(TM) i7-8850H CPU @ 2.60GHz
- 16 GB RAM 2400 MHz DDR4
- Intel UHD Graphics 630 1536 MB
- AMD Radeon Pro RX 560X 4 GB

#### macOS 14.2

- Apple Clang 15.0.0(Integrated in Xcode)
- CMake 3.27.0(Bundled with CLion 2023.3.1)
- OpenCV 4.8.1(Installed by Homebrew)
- OpenCL 1.2(Integrated in Apple Xcode)

## Windows

This program was compiled with 'Microsoft Visual Studio 2022' by default.
You should install Runtime Libraries for 'Microsoft Visual C++' before running them.

You can download them from:
https://learn.microsoft.com/zh-cn/cpp/windows/latest-supported-vc-redist?view=msvc-170

If you download them slow in China, you can use 'Thunder(XunLei)' to download them.

## Linux

### AMD CPU

This program was compiled with 'GNU GCC/G++' by default.
Having a good compatibility with Intel CPU.

You should use 'AMD Optimizing C/C++ and Fortran Compilers (AOCC)' to compile code.
So that you can get more performance on AMD CPU.

You can get more information from:
https://www.amd.com/zh-cn/developer/aocc.html

## macOS

### Intel CPU

This program was compiled with 'Apple Clang' on Intel CPU by default.
So that you can run them directly on macOS.

This program is also compatible with Intel HD Graphics and AMD Radeon GPU on macOS.

### Apple M Series Chip

This program was compiled with 'Apple Clang' on Intel CPU by default.
You should recompile them with 'Apple Clang' on Apple M Series Chip.
So that you get more performance on Apple M Series Chip.
And correctly use Apple GPU.

### Metal 3 API

Please read the following document to get more information about Metal 3 API:
https://support.apple.com/zh-cn/102894