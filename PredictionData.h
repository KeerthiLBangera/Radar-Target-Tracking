#ifndef PREDICTION_DATA_H
#define PREDICTION_DATA_H

#include <vector>
#include <Eigen/Dense>
#include "Track.h"  // Ensure Track.h is included

struct PredictionData {
    bool Is_prediction_done;
    Eigen::VectorXd X_predicted;
    Eigen::MatrixXd P_predicted;
    Eigen::VectorXd z_predicted;
    Eigen::MatrixXd S;
    Eigen::MatrixXd H;

    std::vector<Eigen::Vector2d> Z;  // ✅ Ensure Z is a vector of 2D column vectors
    std::vector<Track> track_list_update;  // Store updated track list

    // ✅ Constructor
    PredictionData()
        : Is_prediction_done(false),
          X_predicted(Eigen::VectorXd::Zero(4)),  
          P_predicted(Eigen::MatrixXd::Zero(4, 4)),
          z_predicted(Eigen::VectorXd::Zero(2)),  
          S(Eigen::MatrixXd::Zero(2, 2)),
          H(Eigen::MatrixXd::Zero(2, 4)),
          Z() {}  // ✅ Initialize empty measurement vector
};

#endif // PREDICTION_DATA_H
