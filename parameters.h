#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <iostream>
#include <vector>
#include <Eigen/Dense> // For matrix operations
#include <cmath>

// ✅ Forward declare Track instead of including "Track.h"
struct Track;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ✅ Structure for a Target
struct Target {
    std::vector<double> time;       // ✅ Added time instances
    Eigen::MatrixXd state;          // ✅ Added state matrix [4 x N]
    std::vector<int> turn;
    double turn_rate;
    Eigen::Vector4d initial_state;
    Eigen::Matrix4d P_initial;
    double start;
    double end;
};

// ✅ Structure for Bias Parameters
struct Bias {
    struct { double scale, offset; } range_bias;
    struct { double scale, offset; } theta_bias;
};

// ✅ Structure for Radar Parameters
struct RadarParameters {
    int NS;
    double T;
    int k_start;
    int k_end;
    double P_D;
    double Lambda;
    double Range_max;
    double Volume_of_meas_space;
    double FA_average;
    Eigen::Matrix2d R;  // Measurement noise covariance
    double P_G;
    double Gating_threshold_meas_space;
};

// ✅ Structure for Filter Parameters
struct FilterParameters {
    double W_min;
    double W_max;
};

// ✅ Structure for Track Maintenance Parameters
struct TrackMaintenance {
    int m1;
    int m2;
    int n;
    double P_birth;
    double P_death;
};

// ✅ Structure for Monte Carlo Simulation Data
struct DataOverMonteCarlo {
    std::vector<std::vector<Eigen::Vector2d>> Z;  // Measurements over scans
    std::vector<Track> track_list_update;         // Track updates
};

// ✅ Main Parameter Data Structure
class Parameter {
public:
    int Monte_Carlo_Runs;
    int scenario;
    double T; // Time step
    int no_of_scans;
    Bias bias;
    RadarParameters radar;
    Eigen::Matrix4d Q_k;  // Process noise covariance
    Eigen::Matrix2d R;    // Measurement noise covariance
    std::vector<Target> targets;
    FilterParameters filter;
    TrackMaintenance track;

    // ✅ Constructor Declaration
    Parameter();

    // ✅ Function Declarations
    void loadParameters();
    Eigen::Matrix4d F_1(double T) const;
    Eigen::Matrix4d F_2(double w, double T) const;
};

// ✅ Function Declaration (Ensure Consistency)
Parameter initialize_parameters();

#endif // PARAMETERS_H
