#ifndef GNN_FILTER_H
#define GNN_FILTER_H

#include <vector>
#include "parameters.h"
#include "Track.h"
#include "Radar.h"
#include "PredictionData.h"

void GNN_filter(std::vector<Track>& track_list, int k, Eigen::MatrixXd Z, 
                Parameter& Parameter, Radar& radar, Eigen::MatrixXd& Z_unassociated);

#endif 