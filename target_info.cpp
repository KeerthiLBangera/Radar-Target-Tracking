#include "target_info.h"
#include "Simulate_target_truth.h"
#include "parameters.h"

#include <vector>
#include <Eigen/Dense>

using Eigen::MatrixXd;

std::vector<Target> target_info(const Parameter& param) {
    std::vector<Target> targets;

    int option = 1;
    int scenario = 1;

    Target t1;
    t1.initial_state = Eigen::Vector4d(10, 30, 10, 50);
    t1.P_initial = Eigen::Matrix4d::Identity() * 10;
    t1.start = 1;
    t1.end = 100;
    Simulate_target_truth(param, t1, option, scenario);
    targets.push_back(t1);

    Target t2;
    t2.initial_state = Eigen::Vector4d(3200, -30, 200, 55);
    t2.P_initial = Eigen::Matrix4d::Identity() * 10;
    t2.start = 1;
    t2.end = 100;
    Simulate_target_truth(param, t2, option, scenario);
    targets.push_back(t2);

    Target t3;
    t3.initial_state = Eigen::Vector4d(500, 30, 220, 55);
    t3.P_initial = Eigen::Matrix4d::Identity() * 10;
    t3.start = 1;
    t3.end = 100;
    Simulate_target_truth(param, t3, option, scenario);
    targets.push_back(t3);

    Target t4;
    t4.initial_state = Eigen::Vector4d(2200,-30, 200, 45);  // [x, vx, y, vy]
    t4.P_initial = Eigen::Matrix4d::Identity() * 10;
    t4.start = 1;
    t4.end = 100;
    Simulate_target_truth(param, t4, option, scenario);
    targets.push_back(t4);

    Target t5;
    t5.initial_state = Eigen::Vector4d(3400, 60, 500, 15);  // [x, vx, y, vy]
    t5.P_initial = Eigen::Matrix4d::Identity() * 10;
    t5.start = 1;
    t5.end = 100;
    Simulate_target_truth(param, t5, option, scenario);
    targets.push_back(t5);

    Target t6;
    t6.initial_state = Eigen::Vector4d(4000, 70, 3000, 20);  // [x, vx, y, vy]
    t6.P_initial = Eigen::Matrix4d::Identity() * 10;
    t6.start = 1;
    t6.end = 100;
    Simulate_target_truth(param, t6, option, scenario);
    targets.push_back(t6);

    Target t7;
    t7.initial_state = Eigen::Vector4d(3200, -30, 2000, 50);  // [x, vx, y, vy]
    t7.P_initial = Eigen::Matrix4d::Identity() * 10;
    t7.start = 1;
    t7.end = 100;
    Simulate_target_truth(param, t7, option, scenario);
    targets.push_back(t7);

    Target t8;
    t8.initial_state = Eigen::Vector4d(6000, 40, 2200, 65);  // [x, vx, y, vy]
    t8.P_initial = Eigen::Matrix4d::Identity() * 10;
    t8.start = 1;
    t8.end = 100;
    Simulate_target_truth(param, t8, option, scenario);
    targets.push_back(t8);

    Target t9;
    t9.initial_state = Eigen::Vector4d(2200, -30, 200, 55);  // [x, vx, y, vy]
    t9.P_initial = Eigen::Matrix4d::Identity() * 10;
    t9.start = 1;
    t9.end = 100;
    Simulate_target_truth(param, t9, option, scenario);
    targets.push_back(t9);

    Target t10;
    t10.initial_state = Eigen::Vector4d(3400, 60, 500, 15);  // [x, vx, y, vy]
    t10.P_initial = Eigen::Matrix4d::Identity() * 10;
    t10.start = 1;
    t10.end = 100;
    Simulate_target_truth(param, t10, option, scenario);
    targets.push_back(t10);

    /*Target t11;
    t11.initial_state = Eigen::Vector4d(6000 ,50 ,3000,30);  // [x, vx, y, vy]
    t11.P_initial = Eigen::Matrix4d::Identity() * 10;
    t11.start = 1;
    t11.end = 100;
    Simulate_target_truth(param, t11, option, scenario);
    targets.push_back(t11);

    Target t12;
    t12.initial_state = Eigen::Vector4d(6400, -30, 4000 ,45);  // [x, vx, y, vy]
    t12.P_initial = Eigen::Matrix4d::Identity() * 10;
    t12.start = 1;
    t12.end = 100;
    Simulate_target_truth(param, t12, option, scenario);
    targets.push_back(t12);

    Target t13;
    t13.initial_state = Eigen::Vector4d(7000 ,30,4000,55);  // [x, vx, y, vy]
    t13.P_initial = Eigen::Matrix4d::Identity() * 10;
    t13.start = 1;
    t13.end = 100;
    Simulate_target_truth(param, t13, option, scenario);
    targets.push_back(t13);

    Target t14;
    t14.initial_state = Eigen::Vector4d(7500, -30 , 4500, 45);  // [x, vx, y, vy]
    t14.P_initial = Eigen::Matrix4d::Identity() * 10;
    t14.start = 1;
    t14.end = 100;
    Simulate_target_truth(param, t14, option, scenario);
    targets.push_back(t14);

    Target t15;
    t15.initial_state = Eigen::Vector4d(8000 ,40 , 5000 ,55);  // [x, vx, y, vy]
    t15.P_initial = Eigen::Matrix4d::Identity() * 10;
    t15.start = 1;
    t15.end = 100;
    Simulate_target_truth(param, t15, option, scenario);
    targets.push_back(t15);

    Target t16;
    t16.initial_state = Eigen::Vector4d(10000, 50 , 5000 ,65);  // [x, vx, y, vy]
    t16.P_initial = Eigen::Matrix4d::Identity() * 10;
    t16.start = 1;
    t16.end = 100;
    Simulate_target_truth(param, t16, option, scenario);
    targets.push_back(t16);

    Target t17;
    t17.initial_state = Eigen::Vector4d(12000, -60 , 6000 ,55);  // [x, vx, y, vy]
    t17.P_initial = Eigen::Matrix4d::Identity() * 10;
    t17.start = 1;
    t17.end = 100;
    Simulate_target_truth(param, t17, option, scenario);
    targets.push_back(t17);

    Target t18;
    t18.initial_state = Eigen::Vector4d(15000,60 ,8000 ,55);  // [x, vx, y, vy]
    t18.P_initial = Eigen::Matrix4d::Identity() * 10;
    t18.start = 1;
    t18.end = 100;
    Simulate_target_truth(param, t18, option, scenario);
    targets.push_back(t18);

    Target t19;
    t19.initial_state = Eigen::Vector4d(16200 , -60 ,12000,45);  // [x, vx, y, vy]
    t19.P_initial = Eigen::Matrix4d::Identity() * 10;
    t19.start = 1;
    t19.end = 100;
    Simulate_target_truth(param, t19, option, scenario);
    targets.push_back(t19);

    Target t20;
    t20.initial_state = Eigen::Vector4d(18400,60, 9000,15);  // [x, vx, y, vy]
    t20.P_initial = Eigen::Matrix4d::Identity() * 10;
    t20.start = 1;
    t20.end = 100;
    Simulate_target_truth(param, t20, option, scenario);
    targets.push_back(t20);
    */

    

    return targets;
}
