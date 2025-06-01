#ifndef MAIN_GNN_H
#define MAIN_GNN_H

#include "parameters.h" 
#include "radar_info.h"
#include "simulate_target_truth.h"
#include "get_measurements_radar.h"
#include "GNN_filter.h"
#include "initialize.h"
#include "track_manager.h"
#include "Radar.h"

#include <vector>
#include <iostream>

// ✅ Corrected MonteCarloData struct
struct MonteCarloData {
    std::vector<Target> X;
    std::vector<std::vector<MeasurementRadar>> Z; // ✅ Fixed Measurement to MeasurementRadar
    std::vector<Track> track_list_update;
};

// ✅ Corrected function prototype to match radar_info.h
std::vector<MonteCarloData> main_GNN(const std::vector<Sensor>& sensors);

#endif // MAIN_GNN_H
