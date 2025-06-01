#ifndef GNN_ASSOCIATION_H
#define GNN_ASSOCIATION_H

#include <vector>
#include <Eigen/Dense>
#include "Track.h"
#include "PredictionData.h"
#include "parameters.h"

// ✅ Function declaration (fixed function signature)
void GNN_association(int k, std::vector<Track>& track_list, 
                     std::vector<PredictionData>& Prediction,
                     Eigen::MatrixXd& Z, int counter, 
                     int No_of_established_tracks, 
                     const Parameter& params,
                     Eigen::MatrixXd& Z_unassociated);

#endif // GNN_ASSOCIATION_H
