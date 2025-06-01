#ifndef RADAR_H
#define RADAR_H

#include <Eigen/Dense>

struct Radar {
    Eigen::VectorXd initial_state;  // [x, vx, y, vy]
    Eigen::MatrixXd P_initial;      // Initial state covariance
    int start;
    int end;

    // Constructor for initialization
    Radar() {
        initial_state = Eigen::VectorXd::Zero(4);  // [0,0,0,0]
        P_initial = Eigen::MatrixXd::Constant(4, 4, NAN); // 4x4 NaN matrix
        start = 1;
        end = 100;
    }
};

#endif // RADAR_H
