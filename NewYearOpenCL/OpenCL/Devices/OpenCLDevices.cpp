//
// Created by konghaomin on 23-12-13.
//

#include <iostream>
#include <vector>

#include "../../Utils/ProgramIO.h"

#include "../Include/OpenCLInclude.h"
#include "OpenCLDevices.h"
#include "OpenCLDevicesList.h"

#include "../../Config/DebugVar.h"

size_t CLGetInfoMaxWorkGroupSize(cl_device_id device) {
    // Query the maximum work group size
    size_t maxWorkGroupSize;
    clGetDeviceInfo(
            device,
            CL_DEVICE_MAX_WORK_GROUP_SIZE,
            sizeof(maxWorkGroupSize),
            &maxWorkGroupSize,
            nullptr
    );

    std::cout << "Max work group size: " << maxWorkGroupSize << std::endl;

    return maxWorkGroupSize;
}

cl_device_id getOpenCLDeviceByIndex(
        const unsigned int platformIndex,
        const unsigned int deviceIndex
) {

    std::cout << "Use Platform " << platformIndex << " Device " << deviceIndex << std::endl;

    // Get available OpenCL platforms
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    // Get count of platforms
    const auto platformSize = platforms.size();

    // Choose the platform (index)
    if (static_cast<unsigned int>(platformSize) <= platformIndex) {
        std::cout << "Error: Platform index out of range!" << std::endl;
        return nullptr;
    }
    cl_platform_id platform = platforms[platformIndex];

    // Get available devices on the platform
    cl_uint numDevices;
    clGetDeviceIDs(
            platform, CL_DEVICE_TYPE_ALL,
            0, nullptr, &numDevices
    );
    std::vector<cl_device_id> devices(numDevices);
    clGetDeviceIDs(
            platform, CL_DEVICE_TYPE_ALL,
            numDevices, devices.data(), nullptr
    );

    // Get count of devices
    const auto deviceSize = devices.size();

    if (static_cast<unsigned int>(deviceSize) <= deviceIndex) {
        std::cout << "Error: Device index out of range!" << std::endl;
        return nullptr;
    }
    // Choose the device (index)
    cl_device_id device = devices[deviceIndex];

    print_multi_char('-', 56);
    std::cout << "Using OpenCL Platform: " << std::endl;

    // Output platform name
    size_t platformNameSize;
    clGetPlatformInfo(
            platform, CL_PLATFORM_NAME, 0,
            nullptr, &platformNameSize
    );
    std::vector<char> platformName(platformNameSize);
    clGetPlatformInfo(
            platform, CL_PLATFORM_NAME, platformNameSize,
            platformName.data(), nullptr
    );
    std::cout << "\tPlatform Name: " << platformName.data() << std::endl;

    std::cout << "Using OpenCL Device: " << std::endl;

    // Output device name
    size_t deviceNameSize;
    clGetDeviceInfo(
            device, CL_DEVICE_NAME,
            0, nullptr, &deviceNameSize
    );
    std::vector<char> deviceName(deviceNameSize);
    clGetDeviceInfo(
            device, CL_DEVICE_NAME,
            deviceNameSize, deviceName.data(), nullptr
    );
    std::cout << "\tDevice Name: " << deviceName.data() << std::endl;

    // Output device vendor
    size_t deviceVendorSize;
    clGetDeviceInfo(
            device, CL_DEVICE_VENDOR,
            0, nullptr, &deviceVendorSize
    );
    std::vector<char> deviceVendor(deviceVendorSize);
    clGetDeviceInfo(
            device, CL_DEVICE_VENDOR,
            deviceVendorSize, deviceVendor.data(), nullptr
    );
    std::cout << "\tDevice Vendor: " << deviceVendor.data() << std::endl;

    // Output device memory
    cl_ulong deviceMemory;
    clGetDeviceInfo(
            device, CL_DEVICE_GLOBAL_MEM_SIZE,
            sizeof(deviceMemory), &deviceMemory, nullptr
    );
    std::cout << "\tDevice Memory: " << deviceMemory / 1024 / 1024 << " MB" << std::endl;

    print_multi_char('-', 56);
//    CLGetInfoMaxWorkGroupSize(device);

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