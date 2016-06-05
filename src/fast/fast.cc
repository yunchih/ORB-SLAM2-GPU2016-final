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

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/core/internal.hpp>

#include "fast.h"

using namespace std;

my::gpu::FAST_GPU::FAST_GPU(int _threshold, bool _nonmaxSuppression, double _keypointsRatio) :
    nonmaxSuppression(_nonmaxSuppression), threshold(_threshold), keypointsRatio(_keypointsRatio), count_(0)
{
}

void my::gpu::FAST_GPU::operator ()(const cv::gpu::GpuMat& image, const cv::gpu::GpuMat& mask, std::vector<cv::KeyPoint>& keypoints, cv::gpu::Stream& stream)
{
    if (image.empty())
        return;

    (*this)(image, mask, d_keypoints_, stream);
    downloadKeypoints(d_keypoints_, keypoints, stream);
}

void my::gpu::FAST_GPU::downloadKeypoints(const cv::gpu::GpuMat& d_keypoints, std::vector<cv::KeyPoint>& keypoints, cv::gpu::Stream& stream)
{
    if (d_keypoints.empty())
        return;

    cv::Mat h_keypoints;
    stream.enqueueDownload(d_keypoints, h_keypoints);
    convertKeypoints(h_keypoints, keypoints);
}

void my::gpu::FAST_GPU::convertKeypoints(const cv::Mat& h_keypoints, std::vector<cv::KeyPoint>& keypoints)
{
    if (h_keypoints.empty())
        return;

    CV_Assert(h_keypoints.rows == ROWS_COUNT && h_keypoints.elemSize() == 4);

    int npoints = h_keypoints.cols;

    keypoints.resize(npoints);

    const short2* loc_row = h_keypoints.ptr<short2>(LOCATION_ROW);
    const float* response_row = h_keypoints.ptr<float>(RESPONSE_ROW);

    for (int i = 0; i < npoints; ++i)
    {
        cv::KeyPoint kp(loc_row[i].x, loc_row[i].y, static_cast<float>(FEATURE_SIZE), -1, response_row[i]);
        keypoints[i] = kp;
    }
}

void my::gpu::FAST_GPU::operator ()(const cv::gpu::GpuMat& img, const cv::gpu::GpuMat& mask, cv::gpu::GpuMat& keypoints, cv::gpu::Stream& stream)
{
    calcKeyPointsLocation(img, mask, stream);
    keypoints.cols = getKeyPoints(keypoints, stream);
}

namespace my { namespace gpu { namespace device
{
    namespace fast
    {
        int calcKeypoints_gpu(PtrStepSzb img, PtrStepSzb mask, short2* kpLoc, int maxKeypoints, PtrStepSzi score, int threshold, cv::gpu::Stream& stream);
        int nonmaxSuppression_gpu(const short2* kpLoc, int count, PtrStepSzi score, short2* loc, float* response, cv::gpu::Stream& stream);
    }
}}}

int my::gpu::FAST_GPU::calcKeyPointsLocation(const cv::gpu::GpuMat& img, const cv::gpu::GpuMat& mask, cv::gpu::Stream& stream)
{
    using namespace my::gpu::device::fast;

    CV_Assert(img.type() == CV_8UC1);
    CV_Assert(mask.empty() || (mask.type() == CV_8UC1 && mask.size() == img.size()));

    int maxKeypoints = static_cast<int>(keypointsRatio * img.size().area());

    ensureSizeIsEnough(1, maxKeypoints, CV_16SC2, kpLoc_);

    if (nonmaxSuppression)
    {
        ensureSizeIsEnough(img.size(), CV_32SC1, score_);
        score_.setTo(cv::Scalar::all(0));
    }

    count_ = calcKeypoints_gpu(img, mask, kpLoc_.ptr<short2>(), maxKeypoints, nonmaxSuppression ? score_ : PtrStepSzi(), threshold, stream);
    count_ = std::min(count_, maxKeypoints);

    return count_;
}

int my::gpu::FAST_GPU::getKeyPoints(cv::gpu::GpuMat& keypoints, cv::gpu::Stream& stream)
{
    using namespace my::gpu::device::fast;

    if (count_ == 0)
        return 0;

    ensureSizeIsEnough(ROWS_COUNT, count_, CV_32FC1, keypoints);

    if (nonmaxSuppression)
        return nonmaxSuppression_gpu(kpLoc_.ptr<short2>(), count_, score_, keypoints.ptr<short2>(LOCATION_ROW), keypoints.ptr<float>(RESPONSE_ROW), stream);

    cv::gpu::GpuMat locRow(1, count_, kpLoc_.type(), keypoints.ptr(0));
    kpLoc_.colRange(0, count_).copyTo(locRow);
    keypoints.row(1).setTo(cv::Scalar::all(0));

    return count_;
}

void my::gpu::FAST_GPU::release()
{
    kpLoc_.release();
    score_.release();

    d_keypoints_.release();
}
