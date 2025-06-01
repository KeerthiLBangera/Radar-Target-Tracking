#ifndef INITIALIZE_H
#define INITIALIZE_H

#include <vector>
#include <Eigen/Dense>
#include "Track.h"
#include "Radar.h"
#include "parameters.h"  // Ensure this matches the correct file name

// Function declaration for initialize
void initialize(const Eigen::MatrixXd& Z_unass, int current_k, 
                const Parameter& param, 
                std::vector<Track>& track_list_update, 
                const Radar& radar);

// Function declaration for initializeTracks
void initializeTracks(std::vector<Eigen::Vector2d>& Z_unassociated, 
                      int k_start, 
                      const Parameter& param, 
                      std::vector<Track>& track_list_update, 
                      const Radar& radar);

#endif // INITIALIZE_H
