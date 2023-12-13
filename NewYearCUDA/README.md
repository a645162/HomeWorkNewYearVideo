# 基于CUDA的新年贺卡视频制作程序

2023年秋季学期 机器视觉(王建华) 课程设计

作者：孔昊旻

## 项目说明

本项目使用CUDA和OpenCV实现了一个简单的新年贺卡视频制作程序，可以生成一个简单的新年贺卡视频。

程序绝大部分功能均使用CUDA实现，使用OpenCV实现的功能仅包括读取图片文件、获取图片尺寸、生成视频。

程序运行过程中，图像数据存储在GPU中，图像处理操作均在GPU上进行，
仅在最终导出为视频过程中涉及CPU与GPU之间的数据传输。
这样大大提升了程序的运行效率，将耗时而复杂的像素处理操作转移到GPU上进行，
使得程序的运行效率得到了极大地提升。

## 开发环境

### CUDA版本说明

因为
CUDA 12 或更新版本才支持40系(Ada Lovelace)显卡，
CUDA 11 或更新版本才支持30系(Ampere)显卡，
因此，我选择CUDA 12以获取最大的兼容性。

但是CUDA12不再支持7系(Kepler)显卡，
Kepler的最新驱动版本为474，
并且CUDA 11的最低驱动要求为450，
因此您可以使用CUDA 11以支持Kepler显卡。

如果您的GPU版本更旧，请您自行选择合适的CUDA版本。

### 参考环境1

#### Hardware

- Intel(R) Xeon(TM) E5-2690 V4 CPU @ 2.60GHz
- NVIDIA Tesla P40 24G(Working on PCIE 3.0 x16)
- DDR4 REG ECC 2400 16G x 4

#### Windows 11 23H2

- Windows 11 23H2
- Visual Studio 2022 17.8.3
- CMake 3.27
- CUDA 12.3.1
- OpenCV 4.8.0
- C++ Standard 20

#### Ubuntu 22.04

- Ubuntu 22.04.3 LTS
- GCC 11.2.0
- CMake 3.22.1
- CUDA 12.3.1
- OpenCV 4.5.3
- C++ Standard 20

### 参考环境2

#### Hardware

- Intel(R) Core(TM) i7-6900K CPU @ 3.20GHz
- NVIDIA GeForce GTX 1080 Ti 11G(Working on PCIE 3.0 x16)
- DDR4 2400 8G x 2

#### Ubuntu 18.04

- Ubuntu 18.04.6 LTS
- GCC 9.4.0
- CMake 3.10.2
- CUDA 12.1
- OpenCV 4.5.4
- C++ Standard 17

## 运行程序

### Windows

在CMakeLists.txt中，将`set(CMAKE_BUILD_TYPE Release)`改为`set(CMAKE_BUILD_TYPE Debug)`，
然后在CMake

## NVIDIA GPU 硬件计算能力参考表

CUDA 12似乎已经不再支持计算能力低于5.0的显卡，
因此如果您的显卡计算能力低于5.0，
您可能需要使用CUDA 11或更低版本。

### GeForce 游戏显卡

- GeForce RTX 40 Series: 8.9
- GeForce RTX 30 Series: 8.6
- GeForce RTX 20 Series: 7.5
- GeForce GTX 16 Series: 7.5
- GeForce GTX 10 Series: 6.1
- GeForce GTX 9 Series: 5.2
- GeForce GTX 8 Series(NoteBook): 5.0
- GeForce GTX 750/750 Ti: 5.0
- GeForce GTX 7 Series: 3.5
- GeForce GTX 6 Series: 3.0

### 其他GPU

- Tesla P100 16G: 6.0
- Tesla P40 24G: 6.1

更多请参考
https://developer.nvidia.com/cuda-gpus

## 项目结构

```
.
├── CMakeLists.txt
├── main.cpp
├── README.md
├── src
│   ├── common.h
│   ├── common.cpp
│   ├── cuda_common.h
│   ├── cuda_common.cpp
│   ├── cuda_image_process.h
```

### 使用OpenCV实现的功能

- 读取图片文件
- 获取图片尺寸
- 生成视频

### 使用CUDA实现的功能

- 图像的旋转
- 图像的缩放
- 图像的裁剪
- 图像的灰度化(平均灰度与加权灰度)
- 渐变颜色生成
- 渐变图像生成
- 图像的高斯模糊
- 图像的高斯滤波
- 图像的遮罩(聚光灯特效)
- 图像的镜像
- 3通道图像转换为4通道图像

## 注意事项

不可以为CUDA开启Fast Math选项， 
否则可能会导致聚光灯效果出现异常。

### VCPkg OpenCV

#### AVI
XVID Fail
MJPG Fail

#### MP4
MP4V Fail
H264
X264
X265
HEVC
AVC1 Fail


### My Build Static OpenCV

#### AVI