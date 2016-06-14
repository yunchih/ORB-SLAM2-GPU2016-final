
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

#### Compute Fast with GPU
The `cv::FAST` function is the main bottleneck in the original implementation.  It is invoked in
every grid of every level of pyramid of every frame to extract their keypoints.  The cost is doubled
if the given grid is of low contrast and `cv::FAST` has to be invoked once morewith a lower threshold.
We reduce the inner level of every level by extracting keypoints of entire frame in one GPU kernel.  In
addition, we enhance data locality by including the low-contrast fallback in the same kernel.

#### Software Pipelining
We use Cuda stream to asynchronously copy memory & launch kernel to hide their latencies, respectively.
In addition, subsequent kernel is launched asynchronously right after the current level, achieving
the pipelining effect analogous to that of hardware.
