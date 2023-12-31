# OpenCL Acceleration

## Hardware Requirement

### CPU

- Intel CPU

### Intel Integrated GPU

For Intel Integrated GPU users,you need to confirm that your CPU support OpenCL 1.2.
The Intel Integrated GPU (Integrated in Intel Core Gen3 Ivy Bridge) is start to support OpenCL 1.2.
So that your CPU must be Intel Core Gen3 Ivy Bridge or newer.
The earliest CPU released in 2012.

### AMD Integrated GPU (APU)

I have not tested on APU yet,so I don't know if it works.
But I think the APU in Ryzen series should work.

### AMD GPU

AMD GPU which architecture is GCN 1.0 or newer is supported for OpenCL 1.2.
For GPU name,you need to confirm that your GPU is "HD 7000 series" or newer.
The earliest GPU which supports OpenCL 1.2 is "HD 7000 series" released in 2012.

### NVIDIA GPU

AMD GPU which architecture is Fermi or newer is supported for OpenCL 1.2.
For GPU name,you need to confirm that your GPU is "GeForce 400 series" or newer.
The earliest GPU which supports OpenCL 1.2 is "GeForce 400 series" released in 2010.

### Intel Isolated GPU (Xe)

I have not tested on Xe yet,so I don't know if it works.
I think it should work.
Because Xe series is released in 2020.

## Compute Backend

### CPU

CPU:pthread

pthread-Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz

Intel(R) Core(TM) i7-8850H CPU @ 2.60GHz

### GPU

- Intel UHD Graphics 630 1536 MB
- AMD Radeon Pro 560X 4 GB
- AMD Radeon Vega 64 8 GB
- NVIDIA Tesla P40 24 GB
- NVIDIA GeForce GTX 1660 Super 6 GB
- NVIDIA P102-100 10 GB

## Feature

- [x] Image Resize
- [x] Image Mask(focus light effect)
- [x] Image Crop
- [x] Image Rotate
- [x] Image Merge
- [x] Channel Convert
- [ ] Image Gray (average gray and weighted gray)
- [ ] Image Mirror
- [x] Gradient Color Generate
- [x] Gradient Image Generate
- [x] Image Rectangle
- [x] Image Convolution
- [x] Image Gaussian Blur
- [x] Image Binarization
- [x] Image Reverse Color