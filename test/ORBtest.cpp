#include <iostream>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>

#include "../include/ORBextractor.h"
#include "../include/csv.hpp"
#include "../include/gms_matcher.h"

using namespace std;

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "./ORBtest PATH_TO_DATASET" << endl;
        return -1;
    }
    std::string image_list_file = std::string(argv[1]) + "data.csv";
    cout << "1 PubImageData start sImage_file: " << image_list_file << endl;
    csv::CSVReader csv_reader(image_list_file);
    std::string image_file;
    cv::Mat image, pre_image;
    auto r = csv_reader.begin();
    for (; r != csv_reader.end(); r++) {
        /* code */
        auto file_name = (*r)["filename"].get<string>();
        file_name = std::string(argv[1]) + "data/" + file_name;
        // image_file = file_name.to_string();
        image = cv::imread(file_name, 0);
        if (pre_image.empty()) {
            pre_image = image;
            continue;
        }
        cv::Mat mask;
        // Load ORB parameters
        cv::Mat descriptors1, descriptors2;
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        int features = 500;
        float scale_factor = 1.2;
        int levels = 8;
        int threshold_FAST = 30;
        int min_threshold_FAST = 7;
        ORB::ORBextractor orb(features, scale_factor, levels, threshold_FAST, min_threshold_FAST);
        orb.ComputeORB(pre_image, mask, keypoints1, descriptors1);
        orb.ComputeORB(image, mask, keypoints2, descriptors2);
        cv::Mat image_show1, image_show2;
        // cv::drawKeypoints(pre_image, keypoints1, image_show1, cv::Scalar(255 * (1 - 0.5), 0, 255 * 0.5));
        // cv::drawKeypoints(image, keypoints1, image_show2, cv::Scalar(255 * (1 - 0.5), 0, 255 * 0.5));

        // cv::imshow("ORB 1", image_show1);
        // cv::imshow("ORB 2", image_show2);
        vector<cv::DMatch> matches;
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
        matcher->match(descriptors1, descriptors2, matches);
        gms_matcher gms(keypoints1, pre_image.size(), keypoints2, image.size(), matches);
        //-- 第四步:匹配点对筛选
        // 计算最小距离和最大距离
        auto min_max = minmax_element(matches.begin(), matches.end(), [](const cv::DMatch& m1, const cv::DMatch& m2) {
            return m1.distance < m2.distance;
        });
        double min_dist = min_max.first->distance;
        //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
        std::vector<cv::DMatch> good_matches, matches_gms;
        for (int i = 0; i < descriptors1.rows; i++) {
            if (matches[i].distance <= max(2 * min_dist, 30.0)) {
                good_matches.push_back(matches[i]);
            }
        }
        std::cout << "number of good match: " << good_matches.size() << std::endl;
        std::vector<uchar> inliers;
        int num_inliners = gms.GetInlierMask(inliers, true, true);
        std::cout << "number of good match by gms: " << num_inliners << std::endl;
        // collect matches
        for (size_t i = 0; i < inliers.size(); ++i) {
            if (inliers[i] == 1) {
                matches_gms.push_back(matches[i]);
            }
        }
        std::vector<int> queryidxs(matches_gms.size()), trainidxs(matches_gms.size());
        for (size_t i = 0; i < matches_gms.size(); i++) {
            queryidxs[i] = matches_gms[i].queryIdx;
            trainidxs[i] = matches_gms[i].trainIdx;
        }
        std::vector<cv::Point2f> points1, points2;
        cv::KeyPoint::convert(keypoints1, points1, queryidxs);
        cv::KeyPoint::convert(keypoints2, points2, trainidxs);

        std::vector<cv::KeyPoint> keypoints_left, keypoints_right;
        cv::KeyPoint::convert(points1, keypoints_left);
        cv::KeyPoint::convert(points2, keypoints_right);
        cv::Mat img_goodmatch, img_gmsmatch;
        // for (size_t i = 0; i < points1.size(); i++) {
        //     circle(pre_image, points1[i], 3, cv::Scalar(100, 200, 100));
        // }
        // for (size_t i = 0; i < points2.size(); i++) {
        //     circle(image, points2[i], 3, cv::Scalar(100, 200, 100));
        // }
        drawMatches(pre_image, keypoints1, image, keypoints2, good_matches, img_goodmatch);
        imshow("matches", img_goodmatch);
        drawMatches(pre_image, keypoints1, image, keypoints2, matches_gms, img_gmsmatch);
        imshow("gms matches", img_gmsmatch);
        pre_image = image;
        cv::waitKey(1);
    }
    return 1;
}