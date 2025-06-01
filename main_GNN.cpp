#include "main_GNN.h"
#include "parameters.h"
#include "radar_info.h"
#include "get_measurements_radar.h"
#include "GNN_filter.h"
#include "initialize.h"
#include "track_manager.h"
#include "Radar.h"
#include "target_info.h"  // ✅ Include your updated target_info

#include <iostream>
#include <vector>

// Helper function to convert measurements to Eigen matrix
Eigen::MatrixXd convertToEigenMatrix(const std::vector<MeasurementRadar>& measurements, size_t scan_index) {
    if (measurements.empty() || scan_index >= measurements.size()) return Eigen::MatrixXd();

    const auto& detections = measurements[scan_index].detections;
    Eigen::MatrixXd matrix(detections.size(), 2);

    for (size_t i = 0; i < detections.size(); ++i) {
        matrix(i, 0) = detections[i][0];  // Range
        matrix(i, 1) = detections[i][1];  // Theta
    }
    return matrix;
}

// Main GNN function
std::vector<MonteCarloData> main_GNN(const std::vector<Sensor>& sensors) {
    Parameter param = initialize_parameters();  
    std::vector<Target> targets = target_info(param);  // ✅ Use updated target_info()

    std::vector<MonteCarloData> Data_over_Monte_Carlo_Runs_MTT(sensors.size());

    // Create scanning time vector
    double scan_time_start = param.radar.k_start;
    double scan_time_end = param.radar.k_end;
    double scan_interval = param.radar.T;
    std::vector<double> scanning_time;

    for (double t = scan_time_start; t <= scan_time_end; t += scan_interval) {
        scanning_time.push_back(t);
    }
    size_t num_scans = scanning_time.size();

    for (size_t SN = 0; SN < sensors.size(); ++SN) {
        std::cout << "Processing Sensor: " << SN + 1 << std::endl;

        Radar radar;
        radar.initial_state = Eigen::VectorXd::Zero(4);         
        radar.P_initial = targets[0].P_initial;                 
        radar.start = param.radar.k_start;
        radar.end = param.radar.k_end;

        std::vector<MonteCarloData> Data_over_Monte_Carlo_Runs(param.Monte_Carlo_Runs);

        for (int MCR = 0; MCR < param.Monte_Carlo_Runs; ++MCR) {
            std::cout << "Monte Carlo Run: " << MCR + 1 << std::endl;

            std::vector<MeasurementRadar> Z = get_measurements_radar(param, scanning_time, sensors, targets);
            std::vector<Track> track_list_update;

            for (size_t k_start = 0; k_start < num_scans; ++k_start) {
                Eigen::MatrixXd Z_matrix = convertToEigenMatrix(Z, k_start);
                Eigen::MatrixXd Z_unassociated = Z_matrix.transpose();  // Transpose to 2xN

                if (!track_list_update.empty()) {
                    Eigen::MatrixXd Z_output_unassociated;
                    GNN_filter(track_list_update, k_start, Z_unassociated, param, radar, Z_output_unassociated);
                    Z_unassociated = Z_output_unassociated;
                }

                initialize(Z_unassociated, k_start, param, track_list_update, radar);
                track_managing(track_list_update, param);
            }

            // Save Monte Carlo data for this run
            Data_over_Monte_Carlo_Runs[MCR].X = targets;
            Data_over_Monte_Carlo_Runs[MCR].Z.push_back(Z);
            Data_over_Monte_Carlo_Runs[MCR].track_list_update = track_list_update;
        }

        // Save data from first Monte Carlo run for simplicity
        Data_over_Monte_Carlo_Runs_MTT[SN].X = targets;
        Data_over_Monte_Carlo_Runs_MTT[SN].Z = Data_over_Monte_Carlo_Runs[0].Z;
        Data_over_Monte_Carlo_Runs_MTT[SN].track_list_update = Data_over_Monte_Carlo_Runs[0].track_list_update;
    }

    return Data_over_Monte_Carlo_Runs_MTT;
}
