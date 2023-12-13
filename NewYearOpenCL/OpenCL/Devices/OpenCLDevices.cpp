//
// Created by konghaomin on 23-12-13.
//

#include <iostream>
#include <vector>

#include "../Include/OpenCLInclude.h"
#include "OpenCLDevices.h"
#include "OpenCLDevicesList.h"

#include "../../Config/DebugVar.h"


cl_device_id getOpenCLDeviceByIndex(const int platformIndex, const int deviceIndex) {

    std::cout << "Use Platform " << platformIndex << " Device " << deviceIndex << std::endl;

    // Get available OpenCL platforms
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    // Get count of platforms
    unsigned int platformSize = platforms.size();

    // Choose the platform (index)
    if (platformSize <= platformIndex) {
        std::cout << "Error: Platform index out of range!" << std::endl;
        return nullptr;
    }
    cl_platform_id platform = platforms[platformIndex];

    // Get available devices on the platform
    cl_uint numDevices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
    std::vector<cl_device_id> devices(numDevices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);

    // Get count of devices
    unsigned int deviceSize = devices.size();

    if (deviceSize <= deviceIndex) {
        std::cout << "Error: Device index out of range!" << std::endl;
        return nullptr;
    }
    // Choose the device (index)
    cl_device_id device = devices[deviceIndex];

    // Output device name
    size_t deviceNameSize;
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize);
    std::vector<char> deviceName(deviceNameSize);
    clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr);
    std::cout << "Using OpenCL device: " << deviceName.data() << std::endl;

    return device;
}

cl_device_id getOpenCLDebugDevice() {
    return getOpenCLDeviceByIndex(DEBUG_CL_PLATFORM_INDEX, DEBUG_CL_DEVICE_INDEX);
}

cl_device_id UserSelectDevice() {
    while (true) {
        auto deviceCount = print_cl_devices_list();
        if (deviceCount == 0) {
            return nullptr;
        }

#ifdef USE_DEBUG_CL_PLATFORM_DEVICE_INDEX
        return getOpenCLDebugDevice();
#else
        int platformIndex, deviceIndex;
        std::cout << "Please select a platform: ";
        std::cin >> platformIndex;
        std::cout << "Please select a device: ";
        std::cin >> deviceIndex;
        if (platformIndex < 0 || deviceIndex < 0) {
            std::cout << "Error: Platform or device index out of range!" << std::endl;
            return nullptr;
        }

        auto device = getOpenCLDeviceByIndex(platformIndex, deviceIndex);
        if (device == nullptr) {
            std::cout << "Error: Platform or device index out of range!" << std::endl;
        } else {
            return device;
        }
#endif
    }
}