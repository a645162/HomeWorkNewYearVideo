# Build Guide

Using Clion to open this project and build it.

## Windows

1. Correctly install GPU (CPU) Driver.
2. clone vcpkg and install OpenCV and OpenCL.

```powershell
git clone https://github.com/microsoft/vcpkg
cd vcpkg
.\bootstrap-vcpkg.bat

.\vcpkg.exe install opencv
.\vcpkg.exe install opencl
```

### Intel CPU Platform Addition Steps

https://software.intel.com/content/www/us/en/develop/articles/opencl-drivers.html

### AMD GPU Platform Addition Steps

https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases

## Linux

```bash
sudo apt install libopencv-dev libpopencl-dev -y
```

## macOS

OpenCL is integrated in Xcode,so you just need to install OpenCV.

```zsh
brew install opencv
```

Notice:You should install Homebrew first.
You can get it from
https://brew.sh/
