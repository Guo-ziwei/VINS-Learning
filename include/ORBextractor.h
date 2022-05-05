/**
 * This file is part of ORB-SLAM2.
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
 * For more information see <https://github.com/raulmur/ORB_SLAM2>
 *
 * ORB-SLAM2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <chrono>
#include <list>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

namespace ORB {
// -------------------------------------------------------------------------------------------------- //

class ExtractorNode {
  public:
    ExtractorNode() : no_more(false) {}
    void divideNode(ExtractorNode& n1, ExtractorNode& n2, ExtractorNode& n3, ExtractorNode& n4);
    bool no_more;
    std::vector<cv::KeyPoint> keys;
    std::list<ExtractorNode>::iterator lit;
    cv::Point2i ul, ur, bl, br;
};

class ORBextractor {
  public:
    enum { HARRIS_SCORE = 0, FAST_SCORE };
    std::vector<cv::Mat> image_pyramid;
    ORBextractor(int _features_num, float _scale_factor, int _levels, int _threshold_FAST, int _min_threshold_FAST);
    ~ORBextractor() = default;
    void ComputeORB(
        cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& _keypoints,
        cv::OutputArray _descriptors);
    void ComputeORB(
        cv::InputArray _image, cv::InputArray _mask, std::vector<cv::Point2f>& _points, cv::OutputArray _descriptors);

  protected:
    int features_num;
    float scalefactor;
    int levels;
    int threshold_FAST, min_threshold_FAST;
    std::vector<cv::Point> pattern;
    std::vector<int> features_per_level;
    std::vector<int> umax;
    std::vector<float> scale_factor;
    std::vector<float> inv_scale_factor;
    std::vector<float> level_sigma2;
    std::vector<float> inv_level_sigma2;
    void computePyramid(cv::Mat image);
    void computeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    std::vector<cv::KeyPoint> distributeOctTree(
        const std::vector<cv::KeyPoint>& distribute_keys, const int& min_x, const int& max_x, const int& min_y,
        const int& max_y, const int& features_num, const int& level);
};

using ORBWrapper = std::shared_ptr<ORBextractor>;

}  // namespace ORB