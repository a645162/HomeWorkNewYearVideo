//
// Created by konghaomin on 2023/12/5.
//

#include "OpenCLDevicesList.h"

#include <iostream>
#include "../../Utils/ProgramIO.h"
#include "../Include/OpenCLInclude.h"
#include "../../Config/Config.h"

unsigned int print_cl_devices_list() {
    print_multi_char('=', 27);
    std::cout << "    OpenCL Devices List\n";
    print_multi_char('=', 27);
    std::cout << "    Author: Haomin Kong" << std::endl;
    print_multi_char('=', 27);
    std::cout << std::endl;

    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);

    auto *platforms = new cl_platform_id[numPlatforms];
    clGetPlatformIDs(numPlatforms, platforms, nullptr);

    unsigned int totalDeviceCount = 0;

    for (cl_uint i = 0; i < numPlatforms; i++) {
        cl_uint numDevices;
        clGetDeviceIDs(
                platforms[i], CL_DEVICE_TYPE_ALL,
                0, nullptr, &numDevices
        );

        auto *devices = new cl_device_id[numDevices];
        clGetDeviceIDs(
                platforms[i], CL_DEVICE_TYPE_ALL,
                numDevices, devices, nullptr
        );

        print_multi_char('-', SEP_LINE_LENGTH);
        std::cout << "Platform " << i << "\n";
        char platformName[1024];
        clGetPlatformInfo(
                platforms[i], CL_PLATFORM_NAME, 1024,
                platformName, nullptr
        );
        std::cout << "\tPlatform Name: " << platformName << "\n";
        std::cout << "\tNumber of Devices: " << numDevices << "\n";
        totalDeviceCount += numDevices;

        for (cl_uint j = 0; j < numDevices; j++) {
            std::cout << "\t";
            print_multi_char('-', 20);
            std::cout << "\tPlatform " << i << " Device " << j << "\n";

            // Device Name
            char deviceName[1024];
            clGetDeviceInfo(
                    devices[j], CL_DEVICE_NAME, sizeof(deviceName),
                    deviceName, nullptr
            );
            std::cout << "\t\tDevice Name: " << deviceName << "\n";

            // Device Vendor
            char vendor[1024];
            clGetDeviceInfo(
                    devices[j], CL_DEVICE_VENDOR, sizeof(vendor),
                    vendor, nullptr
            );
            std::cout << "\t\tVendor Name: " << vendor << "\n";

            // Device Memory Size
            cl_ulong globalMemorySize;
            clGetDeviceInfo(
                    devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
                    &globalMemorySize, nullptr
            );

            std::cout << "\t\tGPU Memory Size: " << globalMemorySize / (1024 * 1024) << " MB" << std::endl;

            // Device Type
            cl_device_type deviceType;
            clGetDeviceInfo(
                    devices[j], CL_DEVICE_TYPE, sizeof(cl_device_type),
                    &deviceType, nullptr
            );
            std::cout << "\t\tDevice Type: ";
            if (deviceType == CL_DEVICE_TYPE_CPU) {
                std::cout << "CPU";
            } else if (deviceType == CL_DEVICE_TYPE_GPU) {
                std::cout << "GPU";
            } else if (deviceType == CL_DEVICE_TYPE_ACCELERATOR) {
                std::cout << "ACCELERATOR";
            } else {
                std::cout << "UNKNOWN";
            }

            std::cout << std::endl;
        }

        delete[] devices;
    }

    if (totalDeviceCount == 0) {
        std::cout << "No OpenCL devices found." << std::endl;
    }

    print_multi_char('-', SEP_LINE_LENGTH);

    delete[] platforms;

    return totalDeviceCount;
}