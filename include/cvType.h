#ifndef __CV_TYPE_HPP__
#define __CV_TYPE_HPP__

#ifndef __NVCC__
typedef struct { short x,y; } short2;
#endif

typedef cv::gpu::PtrStepSz<unsigned char> PtrStepSzb;
typedef cv::gpu::PtrStepSz<float> PtrStepSzf;
typedef cv::gpu::PtrStepSz<int> PtrStepSzi;

#endif
