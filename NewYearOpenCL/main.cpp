//
// Created by konghaomin on 23-12-12.
//

#ifndef DEBUG_MODE
#include "Author/Author.h"
#endif

#include "OpenCL/Devices/OpenCLDevices.h"

#include "Chapter/NewYearCardVideo.h"
#include "Utils/ProgramIO.h"

int main() {
#ifndef DEBUG_MODE
    KHM::sayHello();
#endif

    cl_device_id device = UserSelectDevice();
    cl_context context = CLCreateContext(device);

    // Main Program
    video_main(device, context);

    clReleaseContext(context);
    clReleaseDevice(device);

#ifndef DEBUG_MODE
    WaitForEnterPress();
#endif

    return 0;
}
