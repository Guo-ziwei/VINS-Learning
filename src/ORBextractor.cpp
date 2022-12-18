#include "../include/ORBextractor.h"

namespace ORB {
using namespace std;

constexpr int PATCH_SIZE = 31;
constexpr int HALF_PATCH_SIZE = 15;
constexpr int EDGE_THRESHOLD = 19;

void ExtractorNode::divideNode(ExtractorNode& n1, ExtractorNode& n2, ExtractorNode& n3, ExtractorNode& n4) {
    const int halfX = ceil(static_cast<float>(ur.x - ul.x) / 2);
    const int halfY = ceil(static_cast<float>(br.y - ul.y) / 2);

    // Define boundaries of childs
    n1.ul = ul;
    n1.ur = cv::Point2i(ul.x + halfX, ul.y);
    n1.bl = cv::Point2i(ul.x, ul.y + halfY);
    n1.br = cv::Point2i(ul.x + halfX, ul.y + halfY);
    n1.keys.reserve(keys.size());

    n2.ul = n1.ur;
    n2.ur = ur;
    n2.bl = n1.br;
    n2.br = cv::Point2i(ur.x, ul.y + halfY);
    n2.keys.reserve(keys.size());

    n3.ul = n1.bl;
    n3.ur = n1.br;
    n3.bl = bl;
    n3.br = cv::Point2i(n1.br.x, bl.y);
    n3.keys.reserve(keys.size());

    n4.ul = n3.ur;
    n4.ur = n2.br;
    n4.bl = n3.br;
    n4.br = br;
    n4.keys.reserve(keys.size());

    // Associate points to childs
    for (size_t i = 0; i < keys.size(); i++) {
        const cv::KeyPoint& kp = keys[i];
        if (kp.pt.x < n1.ur.x) {
            if (kp.pt.y < n1.br.y)
                n1.keys.push_back(kp);
            else
                n3.keys.push_back(kp);
        } else if (kp.pt.y < n1.br.y)
            n2.keys.push_back(kp);
        else
            n4.keys.push_back(kp);
    }

    if (n1.keys.size() == 1)
        n1.no_more = true;
    if (n2.keys.size() == 1)
        n2.no_more = true;
    if (n3.keys.size() == 1)
        n3.no_more = true;
    if (n4.keys.size() == 1)
        n4.no_more = true;
}

static float ICAngle(const cv::Mat& image, cv::Point2f pt, const vector<int>& u_max) {
    int m_01 = 0, m_10 = 0;

    const uchar* center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circuI853lar patch
    int step = (int)image.step1();
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v) {
        // Proceed over the two lines
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u) {
            int val_plus = center[u + v * step], val_minus = center[u - v * step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return cv::fastAtan2((float)m_01, (float)m_10);
}

constexpr float factorPI = (float)(CV_PI / 180.0f);
static void computeOrbDescriptor(const cv::KeyPoint& kpt, const cv::Mat& img, const cv::Point* pattern, uchar* desc) {
    float angle = (float)kpt.angle * factorPI;
    float a = (float)cos(angle), b = (float)sin(angle);

    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const int step = (int)img.step;

#define GET_VALUE(idx) \
    center[cvRound(pattern[idx].x * b + pattern[idx].y * a) * step + cvRound(pattern[idx].x * a - pattern[idx].y * b)]

    for (int i = 0; i < 32; ++i, pattern += 16) {
        int t0, t1, val;
        t0 = GET_VALUE(0);
        t1 = GET_VALUE(1);
        val = t0 < t1;
        t0 = GET_VALUE(2);
        t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4);
        t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6);
        t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8);
        t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10);
        t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12);
        t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14);
        t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        desc[i] = (uchar)val;
    }

#undef GET_VALUE
}

static void computeDescriptors(
    const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
    const std::vector<cv::Point>& pattern) {
    descriptors = cv::Mat::zeros(keypoints.size(), 32, CV_8UC1);
    for (size_t i = 0; i < keypoints.size(); i++) {
        /* code */
        computeOrbDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i));
    }
}

ORBextractor::ORBextractor(
    int _features_num, float _scale_factor, int _levels, int _threshold_FAST, int _min_threshold_FAST)
    : features_num(_features_num),
      scalefactor(_scale_factor),
      levels(_levels),
      threshold_FAST(_threshold_FAST),
      min_threshold_FAST(_min_threshold_FAST) {
    scale_factor.resize(levels);
    level_sigma2.resize(levels);
    scale_factor[0] = 1.0f;
    level_sigma2[0] = 1.0f;
    for (int i = 1; i < levels; i++) {
        /* code */
        scale_factor[i] = scale_factor[i - 1] * scalefactor;
        level_sigma2[i] = scale_factor[i] * scale_factor[i];
    }
    inv_scale_factor.resize(levels);
    inv_level_sigma2.resize(levels);
    for (int i = 0; i < levels; i++) {
        /* code */
        inv_scale_factor[i] = 1.0f / scale_factor[i];
        inv_level_sigma2[i] = 1.0f / level_sigma2[i];
    }
    image_pyramid.resize(levels);
    features_per_level.resize(levels);
    float factor = 1.0f / scalefactor;
    float desired_feature_pre_scale = features_num * (1 - factor) / (1 - (float)pow((double)factor, (double)levels));
    int sum_features = 0;
    for (int i = 0; i < levels - 1; i++) {
        /* code */
        features_per_level[i] = cvRound(desired_feature_pre_scale);
        sum_features += features_per_level[i];
        desired_feature_pre_scale *= factor;
    }
    features_per_level[levels - 1] = std::max(features_num - sum_features, 0);
    const int points = 512;
    const cv::Point* pattern0 = (const cv::Point*)ORB_pattern;
    std::copy(pattern0, pattern0 + points, std::back_inserter(pattern));

    // This is for orientation
    // pre-compute the end of a row in a circular patch
    umax.resize(HALF_PATCH_SIZE + 1);
    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));
    // Make sure we are symmetric
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v) {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}

static void computeOrientation(
    const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const std::vector<int>& umax) {
    for (auto& keypoint : keypoints) {
        keypoint.angle = ICAngle(image, keypoint.pt, umax);
    }
}

void ORBextractor::ComputeORB(
    cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& _keypoints, cv::OutputArray _descriptors) {
    if (_image.empty())
        return;
    cv::Mat image = _image.getMat();
    assert(image.type() == CV_8UC1);
    // Pre-compute the scale pyramid
    computePyramid(image);
    std::vector<std::vector<cv::KeyPoint>> allkeypoints;
    computeKeyPointsOctTree(allkeypoints);
    cv::Mat descriptors;
    int keypoints_num = 0;
    for (int i = 0; i < levels; i++) {
        keypoints_num += static_cast<int>(allkeypoints[i].size());
    }
    if (keypoints_num == 0)
        _descriptors.release();
    else {
        _descriptors.create(keypoints_num, 32, CV_8U);
        descriptors = _descriptors.getMat();
    }
    _keypoints.clear();
    _keypoints.reserve(keypoints_num);
    int offset = 0;
    for (int i = 0; i < levels; i++) {
        auto& keypoints = allkeypoints[i];
        if (keypoints.empty())
            continue;
        // preprocess the resized image
        cv::Mat workingMat = image_pyramid[i].clone();
        GaussianBlur(workingMat, workingMat, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
        // Compute the descriptors
        cv::Mat desc = descriptors.rowRange(offset, offset + keypoints.size());
        computeDescriptors(workingMat, keypoints, desc, pattern);

        offset += keypoints.size();
        // Scale keypoint coordinates
        if (i != 0) {
            float scale = scale_factor[i];  // getScale(level, firstLevel, scaleFactor);
            for (auto& keypoint : keypoints)
                keypoint.pt *= scale;
        }
        // And add the keypoints to the output
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }
}

void ORBextractor::computePyramid(cv::Mat image) {
    for (int i = 0; i < levels; i++) {
        /* code */
        float scale = inv_scale_factor[i];
        cv::Size sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));
        cv::Size whole_size(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
        cv::Mat temp(whole_size, image.type()), mask_temp;
        image_pyramid[i] = temp(cv::Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));
        // Compute the resized image
        if (i != 0) {
            cv::resize(image_pyramid[i - 1], image_pyramid[i], sz, 0, 0, cv::INTER_LINEAR);
            cv::copyMakeBorder(
                image_pyramid[i], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                cv::BORDER_REFLECT101 + cv::BORDER_ISOLATED);
        } else {
            copyMakeBorder(
                image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, cv::BORDER_REFLECT_101);
        }
    }
}

std::vector<cv::KeyPoint> ORBextractor::distributeOctTree(
    const std::vector<cv::KeyPoint>& distribute_keys, const int& min_x, const int& max_x, const int& min_y,
    const int& max_y, const int& features_num, const int& level) {
    // Compute how many initial nodes
    const int initial_nodes_num = round(static_cast<float>(max_x - min_x) / (max_y - min_y));
    const float hx = static_cast<float>(max_x - min_x) / initial_nodes_num;
    std::list<ExtractorNode> nodes;
    std::vector<ExtractorNode*> initial_nodes;
    initial_nodes.resize(initial_nodes_num);
    for (int i = 0; i < initial_nodes_num; i++) {
        ExtractorNode ni;
        ni.ul = cv::Point2i(hx * static_cast<float>(i), 0);
        ni.ur = cv::Point2i(hx * static_cast<float>(i + 1), 0);
        ni.bl = cv::Point2i(ni.ul.x, max_y - min_y);
        ni.br = cv::Point2i(ni.ur.x, max_y - min_y);
        ni.keys.reserve(distribute_keys.size());
        nodes.push_back(ni);
        initial_nodes[i] = &nodes.back();
    }
    // Associate points to childs
    for (size_t i = 0; i < distribute_keys.size(); i++) {
        const cv::KeyPoint& kp = distribute_keys[i];
        initial_nodes[static_cast<int>(kp.pt.x / hx)]->keys.push_back(kp);
    }
    auto lit = nodes.begin();
    while (lit != nodes.end()) {
        if (lit->keys.size() == 1) {
            lit->no_more = true;
            lit++;
        } else if (lit->keys.empty()) {
            lit = nodes.erase(lit);
        } else {
            lit++;
        }
    }
    bool is_finish{false};
    int iteration = 0;
    std::vector<pair<int, ExtractorNode*>> size_pointer_2_node;
    size_pointer_2_node.reserve(nodes.size() * 4);
    while (!is_finish) {
        iteration++;
        int prev_size = nodes.size();
        lit = nodes.begin();
        int to_expand = 0;
        size_pointer_2_node.clear();
        while (lit != nodes.end()) {
            if (lit->no_more) {
                // If node only contains one point do not subdivide and continue
                lit++;
                continue;
            } else {
                // If more than one point, subdivide
                ExtractorNode n1, n2, n3, n4;
                lit->divideNode(n1, n2, n3, n4);
                // Add childs if they contain points
                if (n1.keys.size() > 0) {
                    nodes.push_front(n1);
                    if (n1.keys.size() > 1) {
                        to_expand++;
                        size_pointer_2_node.push_back(make_pair(n1.keys.size(), &nodes.front()));
                        nodes.front().lit = nodes.begin();
                    }
                }
                if (n2.keys.size() > 0) {
                    nodes.push_front(n2);
                    if (n2.keys.size() > 1) {
                        to_expand++;
                        size_pointer_2_node.push_back(make_pair(n2.keys.size(), &nodes.front()));
                        nodes.front().lit = nodes.begin();
                    }
                }
                if (n3.keys.size() > 0) {
                    nodes.push_front(n3);
                    if (n3.keys.size() > 1) {
                        to_expand++;
                        size_pointer_2_node.push_back(make_pair(n3.keys.size(), &nodes.front()));
                        nodes.front().lit = nodes.begin();
                    }
                }
                if (n4.keys.size() > 0) {
                    nodes.push_front(n4);
                    if (n4.keys.size() > 1) {
                        to_expand++;
                        size_pointer_2_node.push_back(make_pair(n4.keys.size(), &nodes.front()));
                        nodes.front().lit = nodes.begin();
                    }
                }

                lit = nodes.erase(lit);
                continue;
            }
        }
        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if ((int)nodes.size() >= features_num || (int)nodes.size() == prev_size) {
            is_finish = true;
        } else if (((int)nodes.size() + to_expand * 3) > features_num) {
            while (!is_finish) {
                prev_size = nodes.size();

                vector<pair<int, ExtractorNode*>> prev_size_pointer_2_node = size_pointer_2_node;
                size_pointer_2_node.clear();

                sort(prev_size_pointer_2_node.begin(), prev_size_pointer_2_node.end());
                for (int j = prev_size_pointer_2_node.size() - 1; j >= 0; j--) {
                    ExtractorNode n1, n2, n3, n4;
                    prev_size_pointer_2_node[j].second->divideNode(n1, n2, n3, n4);

                    // Add childs if they contain points
                    if (n1.keys.size() > 0) {
                        nodes.push_front(n1);
                        if (n1.keys.size() > 1) {
                            size_pointer_2_node.push_back(make_pair(n1.keys.size(), &nodes.front()));
                            nodes.front().lit = nodes.begin();
                        }
                    }
                    if (n2.keys.size() > 0) {
                        nodes.push_front(n2);
                        if (n2.keys.size() > 1) {
                            size_pointer_2_node.push_back(make_pair(n2.keys.size(), &nodes.front()));
                            nodes.front().lit = nodes.begin();
                        }
                    }
                    if (n3.keys.size() > 0) {
                        nodes.push_front(n3);
                        if (n3.keys.size() > 1) {
                            size_pointer_2_node.push_back(make_pair(n3.keys.size(), &nodes.front()));
                            nodes.front().lit = nodes.begin();
                        }
                    }
                    if (n4.keys.size() > 0) {
                        nodes.push_front(n4);
                        if (n4.keys.size() > 1) {
                            size_pointer_2_node.push_back(make_pair(n4.keys.size(), &nodes.front()));
                            nodes.front().lit = nodes.begin();
                        }
                    }

                    nodes.erase(prev_size_pointer_2_node[j].second->lit);

                    if ((int)nodes.size() >= features_num)
                        break;
                }

                if ((int)nodes.size() >= features_num || (int)nodes.size() == prev_size)
                    is_finish = true;
            }
        }
    }

    // Retain the best point in each node
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(features_num);
    for (list<ExtractorNode>::iterator lit = nodes.begin(); lit != nodes.end(); lit++) {
        vector<cv::KeyPoint>& vNodeKeys = lit->keys;
        cv::KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        for (size_t k = 1; k < vNodeKeys.size(); k++) {
            if (vNodeKeys[k].response > maxResponse) {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}

void ORBextractor::computeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint>>& allKeypoints) {
    allKeypoints.resize(levels);

    const float W = 30;

    for (int level = 0; level < levels; ++level) {
        const int min_border_x = EDGE_THRESHOLD - 3;
        const int min_border_y = min_border_x;
        const int max_border_x = image_pyramid[level].cols - EDGE_THRESHOLD + 3;
        const int max_border_y = image_pyramid[level].rows - EDGE_THRESHOLD + 3;

        vector<cv::KeyPoint> distribute_keys;
        distribute_keys.reserve(features_num * 10);

        const float width = (max_border_x - min_border_x);
        const float height = (max_border_y - min_border_y);

        const int cols = width / W;
        const int rows = height / W;
        const int cell_w = ceil(width / cols);
        const int cell_h = ceil(height / rows);

        for (int i = 0; i < rows; i++) {
            const float iniY = min_border_y + i * cell_h;
            float maxY = iniY + cell_h + 6;

            if (iniY >= max_border_y - 3)
                continue;
            if (maxY > max_border_y)
                maxY = max_border_y;

            for (int j = 0; j < cols; j++) {
                const float iniX = min_border_x + j * cell_w;
                float maxX = iniX + cell_w + 6;
                if (iniX >= max_border_x - 6)
                    continue;
                if (maxX > max_border_x)
                    maxX = max_border_x;

                vector<cv::KeyPoint> keys_cell;
                FAST(image_pyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX), keys_cell, threshold_FAST, true);
                if (keys_cell.empty()) {
                    FAST(
                        image_pyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX), keys_cell, min_threshold_FAST,
                        true);
                }
                if (!keys_cell.empty()) {
                    for (vector<cv::KeyPoint>::iterator vit = keys_cell.begin(); vit != keys_cell.end(); vit++) {
                        (*vit).pt.x += j * cell_w;
                        (*vit).pt.y += i * cell_h;
                        distribute_keys.push_back(*vit);
                    }
                }
            }
        }

        vector<cv::KeyPoint>& keypoints = allKeypoints[level];
        keypoints.reserve(features_num);

        keypoints = distributeOctTree(
            distribute_keys, min_border_x, max_border_x, min_border_y, max_border_y, features_per_level[level], level);

        const int scaledPatchSize = PATCH_SIZE * scale_factor[level];

        // Add border to coordinates and scale information
        const int nkps = keypoints.size();
        for (int i = 0; i < nkps; i++) {
            keypoints[i].pt.x += min_border_x;
            keypoints[i].pt.y += min_border_y;
            keypoints[i].octave = level;
            keypoints[i].size = scaledPatchSize;
        }
        // cv::Mat debug_image = image_pyramid[level];
        // cv::drawKeypoints(debug_image, keypoints, debug_image,
        // cv::Scalar(255 * (1 - 0.5), 0, 255 * 0.5)); imshow("Fast", debug_image); cv::waitKey(0);
    }

    // compute orientations
    for (int level = 0; level < levels; ++level)
        computeOrientation(image_pyramid[level], allKeypoints[level], umax);
}

}  // namespace ORB