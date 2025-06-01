#include "radar_info.h"
#include <Eigen/Dense>
#include <vector>

std::vector<Sensor> radar_info() {
    std::vector<Sensor> sensors;

    Sensor s1;
    s1.initial_state = Eigen::Vector4d(0, 0, 0, 0);
    s1.P_initial = Eigen::Matrix4d::Constant(NAN);
    s1.start = 1;
    s1.end = 100;
    sensors.push_back(s1);

    Sensor s2;
    s2.initial_state = Eigen::Vector4d(4000, 0, 6000, 0);
    s2.P_initial = Eigen::Matrix4d::Constant(NAN);
    s2.start = 1;
    s2.end = 100;
    sensors.push_back(s2);

    /*
    Sensor s3;
    s3.initial_state = Eigen::Vector4d(700, 6, 0, 2);
    s3.P_initial = Eigen::Matrix4d::Constant(NAN);
    s3.start = 1;
    s3.end = 100;
    sensors.push_back(s3);
    
    // Uncomment and add more sensors as needed
    */

    return sensors;
}
