#ifndef GET_MEASUREMENTS_RADAR_H
#define GET_MEASUREMENTS_RADAR_H

#include <vector>
#include <Eigen/Dense>
#include "parameters.h"     // ✅ For Parameter and Target
#include "radar_info.h"     // ✅ For Sensor structure

// ✅ MeasurementRadar structure for holding radar detections
struct MeasurementRadar {
    std::vector<std::vector<double>> detections;  // [ [range, theta], ... ]
};

// ✅ Function declaration for radar measurement generation
std::vector<MeasurementRadar> get_measurements_radar(
    const Parameter& param,
    const std::vector<double>& scanning_time,
    const std::vector<Sensor>& radars,
    const std::vector<Target>& targets
);

#endif // GET_MEASUREMENTS_RADAR_H