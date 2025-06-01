#include "GNN_association.h"
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <limits>

// ✅ Implementing the GNN_association function
void GNN_association(int k, std::vector<Track>& track_list, 
                     std::vector<PredictionData>& Prediction,
                     Eigen::MatrixXd& Z, int counter, 
                     int No_of_established_tracks, 
                     const Parameter& params,
                     Eigen::MatrixXd& Z_unassociated) {

    if (Z.cols() == 0) return; // Handle empty Z matrix

    int No_of_detections = Z.cols();
    Eigen::MatrixXd Distance_matrix = Eigen::MatrixXd::Constant(counter + 1, No_of_detections + 1, 100);
    std::vector<int> alive_track_index;
    
    for (int i = 0; i < No_of_established_tracks; i++) {
        if (track_list[i].status != -1) { // If track is not dead
            alive_track_index.push_back(i);
            int index_2 = alive_track_index.size(); // Maintain correct index
            for (int j = 1; j <= No_of_detections; j++) {
                Eigen::VectorXd error = Z.col(j - 1) - Prediction[i].z_predicted;
                Distance_matrix(index_2, j) = error.transpose() * Prediction[i].S.inverse() * error;
            }
        }
    }

    Eigen::MatrixXd threshold_constraint_matrix = Eigen::MatrixXd::Ones(counter + 1, No_of_detections + 1);
    threshold_constraint_matrix = (Distance_matrix.array() <= params.radar.Gating_threshold_meas_space)
                                    .select(0, threshold_constraint_matrix);
    threshold_constraint_matrix.col(0).setZero();
    threshold_constraint_matrix.row(0).setZero();

    Eigen::VectorXd cost_vector = Eigen::Map<Eigen::VectorXd>(Distance_matrix.data(), Distance_matrix.size());
    Eigen::VectorXd threshold_vector = Eigen::Map<Eigen::VectorXd>(threshold_constraint_matrix.data(), threshold_constraint_matrix.size());

    Eigen::MatrixXd A_1 = Eigen::MatrixXd::Zero(counter, Distance_matrix.size());
    for (int i = 1; i <= counter; i++) {
        A_1.row(i - 1).segment(i, No_of_detections + 1).setOnes();
    }

    Eigen::MatrixXd A_2 = Eigen::MatrixXd::Zero(No_of_detections, Distance_matrix.size());
    for (int i = 1; i <= No_of_detections; i++) {
        A_2.row(i - 1).segment((counter + 1) * (i - 1), counter + 1).setOnes();
    }

    Eigen::MatrixXd Aeq(A_1.rows() + A_2.rows() + 1, Distance_matrix.size());
    Aeq << A_1, A_2, threshold_vector.transpose();
    Eigen::VectorXd beq = Eigen::VectorXd::Ones(counter + No_of_detections + 1);
    beq(beq.size() - 1) = 0;

    Eigen::VectorXd Assignment_variables_vect = Eigen::VectorXd::Zero(Distance_matrix.size());
    Eigen::MatrixXd Assignment_variables_matrix = Eigen::Map<Eigen::MatrixXd>(Assignment_variables_vect.data(), counter + 1, No_of_detections + 1);
    Eigen::MatrixXd assignment_matrix = Assignment_variables_matrix.block(1, 1, counter, No_of_detections);

    for (int i = 0; i < assignment_matrix.rows(); i++) {
        for (int j = 0; j < assignment_matrix.cols(); j++) {
            if (assignment_matrix(i, j) == 1) {
                int index_1 = alive_track_index[i];

                Eigen::MatrixXd W = Prediction[index_1].P_predicted * Prediction[index_1].H.transpose() * Prediction[index_1].S.inverse();
                Eigen::VectorXd Z_associated = Z.col(j);
                Eigen::VectorXd v = Z_associated - Prediction[index_1].z_predicted;

                Eigen::VectorXd X_updated = Prediction[index_1].X_predicted + W * v;
                Eigen::MatrixXd P_updated = Prediction[index_1].P_predicted - W * Prediction[index_1].S * W.transpose();

                if (!track_list[index_1].track_info.empty()) {
                    auto& last_info = track_list[index_1].track_info.back();

                    last_info.update_time = params.radar.k_start + params.T * (k - 1);
                    last_info.is_associated = true;
                    last_info.x = X_updated;
                    last_info.P = P_updated;
                    last_info.associated_meas = Z_associated;
                    last_info.associated_meas_index = j;
                    last_info.status = last_info.status;

                    // ✅ Fix: Assign the address of Prediction[index_1] instead of the object
                    last_info.prediction = &Prediction[index_1];  
                }

                Z.col(j).setZero(); 
            }
        }
    }

    Z_unassociated = Z;
}
