#include "initialize.h"
#include <iostream>
#include <cmath>  // Ensure math functions work

void initializeTracks(std::vector<Eigen::Vector2d>& Z_unassociated, 
                      int k_start, 
                      const Parameter& param, 
                      std::vector<Track>& track_list_update, 
                      const Radar& radar) {
    // Implementation logic for initializeTracks
}

void initialize(const Eigen::MatrixXd& Z_unass, int current_k, 
                const Parameter& param, std::vector<Track>& track_list_update, 
                const Radar& radar) {
    
    if (Z_unass.cols() > 0) {  // Ensure Z_unass has data
        int list_length = track_list_update.size();

        for (int i = 0; i < Z_unass.cols(); ++i) {
            // Unbiased conversion
            double lambda_beta = std::exp(-param.radar.R(1,1) / 2);  
            double lambda_beta_dash = std::pow(lambda_beta, 4);
            
            double x = (1.0 / lambda_beta) * (Z_unass(0, i) * std::cos(Z_unass(1, i)) + radar.initial_state(0));
            double y = (1.0 / lambda_beta) * (Z_unass(0, i) * std::sin(Z_unass(1, i)) + radar.initial_state(2));

            double R_x = (std::pow(lambda_beta, -2) - 2) * std::pow(Z_unass(0, i), 2) * std::pow(std::cos(Z_unass(1, i)), 2) +
                         0.5 * (std::pow(Z_unass(0, i), 2) + param.radar.R(0,0)) * (1 + lambda_beta_dash * std::cos(2 * Z_unass(1, i)));

            double R_y = (std::pow(lambda_beta, -2) - 2) * std::pow(Z_unass(0, i), 2) * std::pow(std::sin(Z_unass(1, i)), 2) +
                         0.5 * (std::pow(Z_unass(0, i), 2) + param.radar.R(0,0)) * (1 - lambda_beta_dash * std::cos(2 * Z_unass(1, i)));

            double R_xy = (std::pow(lambda_beta, -2) - 2) * std::pow(Z_unass(0, i), 2) * std::cos(Z_unass(1, i)) * std::sin(Z_unass(1, i)) +
                          0.5 * (std::pow(Z_unass(0, i), 2) + param.radar.R(0,0)) * lambda_beta_dash * std::sin(2 * Z_unass(1, i));

            // Track initialization
            Track newTrack;
            newTrack.track_ID = list_length + i + 1;  // 1-based index
            newTrack.status = 0; // Initial

            TrackInfo newTrackInfo;
            newTrackInfo.update_time = current_k;
            newTrackInfo.is_associated = true;
            newTrackInfo.x = Eigen::VectorXd(4);
            newTrackInfo.x << x, 0, y, 0;

            newTrackInfo.P = Eigen::MatrixXd::Zero(4, 4); // Ensure proper initialization
            newTrackInfo.P << R_x, 0, R_xy, 0,
                              0, std::pow(param.radar.Range_max / 2, 2), 0, 0,
                              R_xy, 0, R_y, 0,
                              0, 0, 0, std::pow(param.radar.Range_max / 2, 2);

            newTrackInfo.associated_meas = Z_unass.col(i);
            newTrackInfo.associated_meas_index = i;
            newTrackInfo.status = 0;

            // Add track info to track
            newTrack.track_info.push_back(newTrackInfo);

            // Add track to track list
            track_list_update.push_back(newTrack);
        }
    }
}
