/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#ifndef __MY_OPENCV_FAST_GPU_HPP__
#define __MY_OPENCV_FAST_GPU_HPP__

#include <opencv2/gpu/gpu.hpp>
#include "cvType.h"

namespace my { namespace gpu {

class FAST_GPU
{
public:
    enum
    {
        LOCATION_ROW = 0,
        RESPONSE_ROW,
        ROWS_COUNT
    };

    // all features have same size
    static const int FEATURE_SIZE = 7;

    explicit FAST_GPU(int threshold, bool nonmaxSuppression = true, double keypointsRatio = 0.05);

    //! finds the keypoints using FAST detector
    //! supports only CV_8UC1 images
    void operator ()(const cv::gpu::GpuMat& image, const cv::gpu::GpuMat& mask, cv::gpu::GpuMat& keypoints, cv::gpu::Stream& stream);
    void operator ()(const cv::gpu::GpuMat& image, const cv::gpu::GpuMat& mask, std::vector<cv::KeyPoint>& keypoints, cv::gpu::Stream& stream);

    //! download keypoints from device to host memory
    void downloadKeypoints(const cv::gpu::GpuMat& d_keypoints, std::vector<cv::KeyPoint>& keypoints, cv::gpu::Stream& stream);

    //! convert keypoints to cv::KeyPoint vector
    void convertKeypoints(const cv::Mat& h_keypoints, std::vector<cv::KeyPoint>& keypoints);

    //! release temporary buffer's memory
    void release();

    bool nonmaxSuppression;

    int threshold;

    //! max keypoints = keypointsRatio * img.size().area()
    double keypointsRatio;

    //! find keypoints and compute it's response if nonmaxSuppression is true
    //! return count of detected keypoints
    int calcKeyPointsLocation(const cv::gpu::GpuMat& image, const cv::gpu::GpuMat& mask, cv::gpu::Stream& stream);

    //! get final array of keypoints
    //! performs nonmax suppression if needed
    //! return final count of keypoints
    int getKeyPoints(cv::gpu::GpuMat& keypoints, cv::gpu::Stream& stream);

private:
    cv::gpu::GpuMat kpLoc_;
    int count_;

    cv::gpu::GpuMat score_;

    cv::gpu::GpuMat d_keypoints_;
};

} // namespace: gpu

} // namespace: my

#endif
