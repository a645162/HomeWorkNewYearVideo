# OpenCL Driver Guide

Last Updated: 2023-12-12
Author: Haomin Kong

## Windows

### Intel CPU

We have two solutions to install Intel CPU OpenCL Driver on Windows.

1. Install Intel CPU OpenCL Driver

You can download the driver from:
https://www.intel.cn/content/www/cn/zh/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html

OR

2. Install Intel oneAPI.

### GPU

If you install GPU Driver correctly, you can use GPU in OpenCL without any other configuration.

## Ubuntu

```bash
sudo apt install opencl-dev -y
```

### NVIDIA GPU

Install NVIDIA Driver.

```bash
# install NVIDIA Driver(Please replace 545 with your driver version)
sudo apt install nvidia-driver-545
```

If you can't use NVIDIA Devices, you can try to install CUDA.

### AMD GPU

You need to install 'amdgpu-install' first refer to:
https://rocm.docs.amd.com/en/latest/deploy/linux/installer/install.html

You only need to install the AMD GPU Driver and OpenCL SDK, so you can use this command:

```bash
sudo amdgpu-install --usecase=graphics,openclsdk
```

Notice:You can use 'workstation' to replace open source driver 'graphics' if you need more features.

If you can't use AMD Devices without root permission, you can try this:

```bash
sudo usermod -a -G video $LOGNAME
sudo usermod -a -G render $LOGNAME
```

### Intel GPU (Include Intel HD Graphics)

For Intel Integrated GPU users,you need to confirm that your CPU support OpenCL 1.2.
The Intel Integrated GPU (Integrated in Intel Core Gen3 Ivy Bridge) is start to support OpenCL 1.2.
So that your CPU must be Intel Core Gen3 Ivy Bridge or newer.

```bash
# This software package is already included in the Ubuntu source.
sudo apt install intel-opencl-icd
```

### Intel CPU

https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2023-0/apt.html

```bash
# download the key to system keyring
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
| gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

# add signed entry to apt sources and configure the APT client to use Intel repository:
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

# update the APT cache
sudo apt update

# install the runtime
sudo apt install intel-oneapi-runtime-opencl
```

## Reference

https://cn.linux-console.net/?p=19885