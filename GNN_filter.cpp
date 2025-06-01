#include "GNN_filter.h"
#include "GNN_association.h"
#include <iostream>
#include <cmath>

void GNN_filter(std::vector<Track>& track_list, int k, Eigen::MatrixXd Z, 
                Parameter& Parameter, Radar& radar, Eigen::MatrixXd& Z_unassociated) {
    
    int counter = 0;
    int No_of_established_tracks = track_list.size();
    std::vector<PredictionData> Prediction(No_of_established_tracks);

    for (int i = 0; i < No_of_established_tracks; ++i) {
        if (track_list[i].status != -1) {
            counter++;

            Eigen::Vector4d X_predicted = Parameter.F_1(Parameter.T) * track_list[i].track_info.back().x;
            Eigen::Matrix4d P_predicted = Parameter.F_1(Parameter.T) * track_list[i].track_info.back().P * Parameter.F_1(Parameter.T).transpose() + Parameter.Q_k;

            double range_predicted = sqrt(pow(X_predicted(0) - radar.initial_state(0), 2) + pow(X_predicted(2) - radar.initial_state(2), 2));
            double theta_predicted = atan2(X_predicted(2) - radar.initial_state(2), X_predicted(0) - radar.initial_state(0));
            Eigen::Vector2d z_predicted(range_predicted, theta_predicted);

            Eigen::Matrix<double, 2, 4> H = Eigen::Matrix<double, 2, 4>::Zero();
            H(0, 0) = (X_predicted(0) - radar.initial_state(0)) / range_predicted;
            H(0, 2) = (X_predicted(2) - radar.initial_state(2)) / range_predicted;
            H(1, 0) = -(X_predicted(2) - radar.initial_state(2)) / (pow(X_predicted(0) - radar.initial_state(0), 2) + pow(X_predicted(2) - radar.initial_state(2), 2));
            H(1, 2) = 1.0 / (X_predicted(0) - radar.initial_state(0));

            Eigen::MatrixXd HPHT = H * P_predicted * H.transpose();

            if (Parameter.R.rows() != HPHT.rows() || Parameter.R.cols() != HPHT.cols()) {
                std::cerr << "[Fatal] Matrix size mismatch in residual covariance computation:\n";
                std::cerr << "Parameter.R size: " << Parameter.R.rows() << "x" << Parameter.R.cols() << "\n";
                std::cerr << "HPHT size: " << HPHT.rows() << "x" << HPHT.cols() << "\n";

                Parameter.R = Eigen::Matrix2d::Zero();
                Parameter.R(0, 0) = 900;
                Parameter.R(1, 1) = 0.0009;
                std::cerr << "[Fix] Reinitialized Parameter.R to default 2x2 matrix.\n";
            }

            Eigen::Matrix2d S = Parameter.R + HPHT;

            Prediction[i].z_predicted = z_predicted;
            Prediction[i].S = S;
            Prediction[i].H = H;
            Prediction[i].X_predicted = X_predicted;
            Prediction[i].P_predicted = P_predicted;

            TrackInfo newTrackInfo;
            newTrackInfo.update_time = k;
            newTrackInfo.is_associated = false;
            newTrackInfo.x = X_predicted;
            newTrackInfo.P = P_predicted;
            newTrackInfo.z_predicted = z_predicted;
            newTrackInfo.S = S;
            newTrackInfo.status = track_list[i].track_info.back().status;

            // ✅ Set track_ID for visualization grouping
            //newTrackInfo.track_ID = track_list[i].track_ID;

            track_list[i].track_info.push_back(newTrackInfo);
        }
    }

    if (Z.size() > 0) {
        GNN_association(k, track_list, Prediction, Z, counter, No_of_established_tracks, Parameter, Z_unassociated);
    } else {
        Z_unassociated = Z;
    }
}
