#pragma once

#include "../utility/utility.h"
#include "../parameters.h"

#include <ceres/ceres.h>
using namespace Eigen;

class IntegrationBaseCpi
{
public:
    IntegrationBaseCpi() = delete;
    IntegrationBaseCpi(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                    const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg ) 
                    : acc_0{_acc_0}, gyr_0{_gyr_0}, linearized_acc{_acc_0}, linearized_gyr{_gyr_0},
          linearized_ba{_linearized_ba}, linearized_bg{_linearized_bg},
            jacobian{Eigen::Matrix<double, 15, 15>::Identity()}, covariance{Eigen::Matrix<double, 15, 15>::Zero()},
          sum_dt{0.0}, delta_p{Eigen::Vector3d::Zero()}, delta_q{Eigen::Quaterniond::Identity()}, delta_v{Eigen::Vector3d::Zero()}
    {
        noise = Eigen::Matrix<double, 18, 18>::Zero();
        noise.block<3,3>(0, 0) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3,3>(3, 3) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3,3>(6, 6) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3,3>(9, 9) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
    }

    void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
    {
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        propagate(dt, acc, gyr);
    }

    void propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1)
    {
        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;
        sum_dt += dt;
        // get angle change w*dt
        Eigen::Vector3d w_hat = gyr_1 - linearized_bg;
        Eigen::Vector3d a_hat = acc_1 - linearized_ba;
        Eigen::Vector3d w_hatdt = w_hat * dt;
        // Threshold to determine if equations will be unstable
        double w_hat_norm = w_hat.norm();
        bool small_w = (w_hat_norm < 0.008726646);
        double w_dt = w_hat_norm * dt;
        double dt_2 = dt * dt;
        double cos_wt = cos(w_dt);
        double sin_wt = sin(w_dt);
        Eigen::Matrix3d w_x = skew(w_hat);
        Eigen::Matrix3d a_w = skew(a_hat);
        Eigen::Matrix3d w_tx = skew(w_hatdt);
        Eigen::Matrix3d w_x_2 = w_x * w_x;
        //get relative rotation
        Eigen::Vector3d dR = small_w ? eye3 - dt * w_x + dt_2 / 2 * w_x_2 :
                                                    eye3 - (sin_wt / w_hat_norm) * w_x + (1.0 - cos_wt) / (w_hat_norm*w_hat_norm) * w_x_2;
        dR = dR.transpose().eval();
        delta_q = Eigen::Quaterniond(dR);
        double f1, f2, f3, f4;
        if (small_w) {
            f1 = -pow(dt, 3) / 3;
            f2 = pow(dt, 4) / 8;
            f3 = -dt_2 / 2;
            f4 = pow(dt, 3) / 6;
        } else {
            f1 = (w_dt * cos_wt - sin_wt) / pow(w_hat_norm, 3);
            f2 = (w_dt * w_dt - 2 * cos_wt - 2 * w_dt * sin_wt + 2) / (2 * pow(w_hat_norm, 4));
            f3 = -(1 - cos_wt) / pow(w_hat_norm, 2);
            f4 = (w_dt - sin_wt) / pow(w_hat_norm, 3);
        }
        delta_p = delta_v * dt + (delta_q * ((dt_2 / 2) * eye3) + f1 * w_x + f2 * w_x_2) * a_hat; // 需要验证delta_q * ((dt_2 / 2) * eye3是否正确
        delta_v = dR * (dt * eye3 + f3 * w_x + f4 * w_x_2) * a_hat;
        
        MatrixXd F = MatrixXd::Zero(15, 15);
        MatrixXd G = MatrixXd::Zero(15, 12);
        F.block<3,3>(0, 6) = Eigen::Matrix3d::Identity();
        F.block<3,3>(3, 3) = - w_x;
        F.block<3,3>(3, 12) = -Eigen::Matrix3d::Identity();
        F.block<3,3>(6, 3) = dR * a_w;
        F.block<3,3>(6, 6) = dR;

        G.block<3,3>(0, 0) = -dR;
        G.block<3,3>(3, 3) = -eye3;
        G.block<3,3>(6, 6) = eye3;
        G.block<3,3>(9, 9) = eye3;

        F = Eigen::Matrix<double, 15, 15>::Identity() + dt * F;
        G *= dt;

        jacobian = F * jacobian;
        covariance = F * covariance * F.transpose() + G * noise * G.transpose();
    }

    void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
    {
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;
        jacobian.setIdentity();
        covariance.setZero();
        for (size_t i = 0; i < dt_buf.size(); i++)
        {
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
        }
    }

    Eigen::Matrix<double, 15, 1> evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                                          const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj)
    {
        Eigen::Matrix<double, 15, 1> residual;
        Eigen::Matrix3d dp_dba = ;
        Eigen::Matrix3d dp_dbg = ;
        return residual;
    }

    double dt;
    Eigen::Vector3d acc_0, gyr_0;
    Eigen::Vector3d acc_1, gyr_1;

    const Eigen::Vector3d linearized_acc, linearized_gyr;
    Eigen::Vector3d linearized_ba, linearized_bg;

    Eigen::Matrix<double, 15, 15> jacobian, covariance;
    Eigen::Matrix<double, 15, 15> step_jacobian;
    Eigen::Matrix<double, 15, 18> step_V;
    Eigen::Matrix<double, 12, 12> noise;
    Eigen::Matrix3d eye3 = Eigen::Matrix3d::Identity();

    double sum_dt;
    Eigen::Vector3d delta_p;
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_v;

    std::vector<double> dt_buf;
    std::vector<Eigen::Vector3d> acc_buf;
    std::vector<Eigen::Vector3d> gyr_buf;

private:
    Eigen::Matrix3d skew(const Eigen::Vector3d &vec3_)
    {
        Eigen::Matrix3d skew_x = Eigen::Matrix3d::Zero();
        skew_x(0,1) = -vec3_(2);
        skew_x(1,0) = vec3_(2);
        skew_x(0,2) = vec3_(1);
        skew_x(2,0) = vec3_(1);
        skew_x(1,2) = vec3_(0);
        skew_x(2,1) = vec3_(0);
        return skew_x;
    }
};

