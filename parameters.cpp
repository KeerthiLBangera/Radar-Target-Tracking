#include "parameters.h"
#include "target_info.h"  // ✅ For target_info()
#include <iostream>
#include <cmath>

// ✅ Constructor definition
Parameter::Parameter()
    : Monte_Carlo_Runs(1), scenario(1), no_of_scans(50) {

    // ✅ Radar Parameters Initialization
    radar.NS = 1;
    radar.T = 1.0;
    radar.k_start = 1;
    radar.k_end = 100;
    radar.P_D = 0.7;
    radar.Lambda = 1e-5;
    radar.Range_max = 10000;
    radar.Volume_of_meas_space = 2 * M_PI * radar.Range_max;
    radar.FA_average = radar.Lambda * radar.Volume_of_meas_space;

    radar.R = Eigen::Matrix2d::Zero(); 
    radar.R(0, 0) = 900;     // Variance for range
    radar.R(1, 1) = 0.0009;  // Variance for angle
    radar.P_G = 0.99999;
    radar.Gating_threshold_meas_space = 23.2093;

    // ✅ Bias Parameters
    bias.range_bias = {1e-4, 5};
    bias.theta_bias = {1e-6, 0.015};

    // ✅ Target Process Noise Initialization
    Q_k = Eigen::Matrix4d::Identity() * 0.01;

    // ✅ Filter Parameters
    filter.W_min = 0.1 * M_PI / 180;
    filter.W_max = 15 * M_PI / 180;

    // ✅ Track Maintenance Parameters
    track.m1 = 7;
    track.m2 = 4;
    track.n = 10;
    track.P_birth = 0.8;
    track.P_death = 0.2;
}

// ✅ Motion Model for Constant Velocity
Eigen::Matrix4d Parameter::F_1(double T) const {
    Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
    F(0, 1) = T;
    F(2, 3) = T;
    return F;
}

// ✅ Motion Model for Constant Turn Rate
Eigen::Matrix4d Parameter::F_2(double w, double T) const {
    Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
    if (w != 0) {
        F << 1, sin(w * T) / w, 0, -(1 - cos(w * T)) / w,
             0, cos(w * T), 0, -sin(w * T),
             0, (1 - cos(w * T)) / w, 1, sin(w * T) / w,
             0, sin(w * T), 0, cos(w * T);
    }
    return F;
}

// ✅ Function to Initialize and Return Parameters
Parameter initialize_parameters() {
    Parameter param;

    // Copy radar.R to top-level R for measurement model
    param.R = param.radar.R;

    // ✅ Load target info from target_info.cpp
    //param.targets = target_info();
   param.targets = target_info(param);

    // ✅ Full turn data for 20 targets
    std::vector<int> turn_start = {
        151, 120, 120, 120, 120,
        151, 120, 120, 120, 120,
        151, 120, 120, 120, 120,
        151, 120, 120, 120, 120
    };

    std::vector<int> turn_end = {
        170, 160, 160, 160, 160,
        170, 160, 160, 160, 160,
        170, 160, 160, 160, 160,
        170, 160, 160, 160, 160
    };

    std::vector<double> turn_rates = {
        -0.5 * M_PI / 180, 0.5 * M_PI / 180, 0.5 * M_PI / 180, 0.5 * M_PI / 180, 0.5 * M_PI / 180,
        -0.5 * M_PI / 180, 0.5 * M_PI / 180, 0.5 * M_PI / 180, 0.5 * M_PI / 180, 0.5 * M_PI / 180,
        -0.5 * M_PI / 180, 0.5 * M_PI / 180, 0.5 * M_PI / 180, 0.5 * M_PI / 180, 0.5 * M_PI / 180,
        -0.5 * M_PI / 180, 0.5 * M_PI / 180, 0.5 * M_PI / 180, 0.5 * M_PI / 180, 0.5 * M_PI / 180
    };

    for (size_t i = 0; i < std::min({param.targets.size(), turn_start.size(), turn_rates.size()}); ++i) {
        for (int k = turn_start[i]; k <= turn_end[i]; ++k) {
            param.targets[i].turn.push_back(k);
        }
        param.targets[i].turn_rate = turn_rates[i];
    }

    return param;
}

// ✅ Load Parameters from mock config
void Parameter::loadParameters() {
    Monte_Carlo_Runs = 10;
    no_of_scans = 200;
    radar.k_end = 150;
    std::cout << "Parameters loaded successfully." << std::endl;
}
