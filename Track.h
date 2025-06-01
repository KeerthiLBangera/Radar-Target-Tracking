#ifndef TRACK_H
#define TRACK_H

#include <Eigen/Dense>
#include <vector>

// Forward declaration instead of #include
struct PredictionData; 

// Structure to store track information
struct TrackInfo {
    //int track_ID = -1;  // ✅ Added: Unique ID to track the parent track across time

    int update_time = 1;
    bool is_associated = true;
    Eigen::VectorXd x = (Eigen::VectorXd(4) << -0.0790757381745263, 0, 18.1227092259820, 0).finished();
    Eigen::MatrixXd P = (Eigen::MatrixXd(4, 4) << 
        1.12143303966328, 0, -3.91859313678900, 0,
        0, 4225, 0, 0,
        -3.91859313678900, 0, 899.174028940898, 0,
        0, 0, 0, 4225).finished();
    Eigen::VectorXd associated_meas = (Eigen::VectorXd(2) << 18.1147, 1.5752).finished();
    int associated_meas_index = 1;
    int status = 0;

    PredictionData* prediction;  // Use pointer instead of direct object
    Eigen::VectorXd z_predicted = Eigen::VectorXd::Zero(2);
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(2, 2);

    TrackInfo()
        : update_time(0), 
          is_associated(false), 
          x(Eigen::VectorXd::Zero(4)), 
          P(Eigen::MatrixXd::Zero(4, 4)), 
          z_predicted(Eigen::VectorXd::Zero(2)), 
          S(Eigen::MatrixXd::Zero(2, 2)), 
          status(0),
          prediction(nullptr) {}  // Initialize to null
};

// Structure for a track
struct Track {
    int track_ID;
    int status;
    std::vector<TrackInfo> track_info;
    Eigen::VectorXd state;  // Track state vector
    Eigen::MatrixXd P;      // Covariance matrix

    // Constructor defined inside header
    Track() : track_ID(1), status(0), track_info({TrackInfo()}),
              state(Eigen::VectorXd::Zero(4)),
              P(Eigen::MatrixXd::Identity(4, 4)) {}

    void predict(double dt);
    void update(const Eigen::VectorXd& measurement);
    bool isLost() const;
};

// Predict function (can be defined inline if needed)
inline void Track::predict(double dt) {
    // Example prediction logic (modify as needed)
}

// Update function (optional, modify as needed)
inline void Track::update(const Eigen::VectorXd& measurement) {
    // Example update logic
}

// Check if the track is lost
inline bool Track::isLost() const {
    return status == -1;
}

#endif // TRACK_H
