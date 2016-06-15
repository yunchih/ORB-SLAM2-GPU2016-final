
# ORB-SLAM2 GPU optimization
## GPUGPU 2016 Final Project


### Introduction

### Video

[![Project Proposal](https://thumbnail.jpg)](https://www.youtube.com/watch?v=ID_HERE)
[![Project Demo](https://thumbnail.jpg)](https://www.youtube.com/watch?v=ID_HERE)

### Optimization detail

##### Switch from OpenCV 2.4 to OpenCV 3.1
OpenCV 3.1 introduces several features helpful to this project: custom memory allocator, 
Cuda stream and rewrite of some essential algorithms, such as Fast and ORB.
These features allow us to fully exploit more Cuda APIs, such as Unified Memory.

##### Compute Fast with GPU
The `cv::FAST` function is the main bottleneck in the original implementation.  It is invoked in
every grid of every level of pyramid of every frame to extract their keypoints.  The cost is doubled
if the given grid is of low contrast and `cv::FAST` has to be invoked once more with a lower threshold.
We reduce the inner level of every level by extracting keypoints of entire frame in one GPU kernel.  In
addition, we enhance data locality by including the low-contrast fallback in the same kernel.

##### Software Pipelining
We use Cuda stream to asynchronously copy memory & launch kernel to hide their latencies, respectively.
In addition, kernel of subsequent level is launched asynchronously right after current level, achieving
the pipelining effect analogous to that of hardware.  The technique is used in both extracting
FAST keypoints and computing ORB descriptors.

##### Unified Memory
We use Cuda Unified Memory, exploiting the unique unified physical memory of Nvidia Tegra, to save host-device
memory copy.

##### Save memory allocation/free
In the original implementation, some frequently invoked functions locally allocate/free memory in every invocation.
We save such overhead by allocating memory once and free it when it will no longer be used.

##### Transform still more procedures into GPU
Functions such as `resize`, `copyMakeBorder`, `Gaussian filter` are directly replaced by the their GPU counterparts
in OpenCV.  Functions such as `compute_IC_Angle`, `compute_ORB_Descriptor` are adopted from OpenCV's
implementation to fit our needs.
