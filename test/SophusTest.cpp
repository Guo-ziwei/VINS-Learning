#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <sophus/so3.hpp>

using namespace Sophus;

int main() {
    Eigen::Vector3d omega(0.02, 0.01, 0.01);
    Eigen::Matrix3d Jr = SO3d::leftJacobian(omega).transpose();
    Eigen::Matrix3d JrInv = SO3d::rightJacobianInverse(omega);
    Eigen::Matrix3d JrInv_ = SO3d::leftJacobianInverse(omega).transpose();
    Eigen::Matrix3d product1 = Jr * JrInv;
    Eigen::Matrix3d product2 = JrInv * Jr;
    Eigen::Matrix3d product3 = JrInv_ * Jr;
    double err1 = (product1 - Eigen::Matrix3d::Identity()).norm();
    double err2 = (product2 - Eigen::Matrix3d::Identity()).norm();
    double err3 = (product3 - Eigen::Matrix3d::Identity()).norm();
    std::cout << "JrInv_ = " << JrInv_ << std::endl;
    std::cout << "Jr = " << Jr << std::endl;
    std::cout << "JrInv = " << JrInv << std::endl;
    std::cout << "Test 2 (small omega): |Jr*JrInv - I| = " << err1;
    std::cout << (err1 < 1e-6 ? " [PASS]" : " [FAIL]") << std::endl;
    std::cout << "                      |JrInv*Jr - I| = " << err2;
    std::cout << (err2 < 1e-6 ? " [PASS]" : " [FAIL]") << std::endl;
    std::cout << "Jr*JrInv_ = " << Jr * JrInv_ << std::endl;
    std::cout << "JrInv_*Jr = " << JrInv_ * Jr << std::endl;
    std::cout << "                      |Jr*JrInv_ - I| = " << err3;
    std::cout << (err3 < 1e-6 ? " [PASS]" : " [FAIL]") << std::endl;
    return 0;
}