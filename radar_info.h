#ifndef RADAR_INFO_H
#define RADAR_INFO_H

#include <vector>
#include <Eigen/Dense>

struct Sensor {
    Eigen::Vector4d initial_state;
    Eigen::Matrix4d P_initial;
    int start;
    int end;
};

std::vector<Sensor> radar_info();

#endif // RADAR_INFO_H
