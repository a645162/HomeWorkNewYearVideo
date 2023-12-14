//
// Created by konghaomin on 23-12-13.
//

#ifndef NEW_YEAR_OPENCL_OPENCL_DEVICES_H
#define NEW_YEAR_OPENCL_OPENCL_DEVICES_H

size_t CLGetInfoMaxWorkGroupSize(cl_device_id device);

cl_device_id getOpenCLDeviceByIndex(int platformIndex, int deviceIndex);

cl_device_id UserSelectDevice();

#endif //NEW_YEAR_OPENCL_OPENCL_DEVICES_H
