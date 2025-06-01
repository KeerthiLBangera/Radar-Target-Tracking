#include "get_measurements_radar.h"
#include <cmath>
#include <algorithm>
#include <random>

std::vector<MeasurementRadar> get_measurements_radar(
    const Parameter& param,
    const std::vector<double>& scanning_time,
    const std::vector<Sensor>& radars,
    const std::vector<Target>& targets
) {
    size_t no_of_scans = scanning_time.size();
    size_t no_of_targets = targets.size();
    std::vector<MeasurementRadar> Z(no_of_scans);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::poisson_distribution<int> poisson_FA(param.radar.FA_average);  // Still initialized in case you re-enable
    std::uniform_real_distribution<double> uniform_range(0, param.radar.Range_max);
    std::uniform_real_distribution<double> uniform_theta(-M_PI, M_PI);
    std::normal_distribution<double> normal_range(0, sqrt(param.radar.R(0, 0)));
    std::normal_distribution<double> normal_theta(0, sqrt(param.radar.R(1, 1)));

    for (size_t n = 0; n < no_of_scans; ++n) {
        std::vector<std::vector<double>> detections;

        // Add true detections with measurement noise
        for (size_t m = 0; m < no_of_targets; ++m) {
            auto it = std::find(targets[m].time.begin(), targets[m].time.end(), scanning_time[n]);
            if (it != targets[m].time.end() && ((double)rand() / RAND_MAX) <= param.radar.P_D) {
                size_t index = std::distance(targets[m].time.begin(), it);

                // True range and theta based on target states and radar position
                double range = std::sqrt(
                    std::pow(targets[m].state(0, index) - radars[0].initial_state(0), 2) +
                    std::pow(targets[m].state(2, index) - radars[0].initial_state(2), 2)
                );

                double theta = std::atan2(
                    targets[m].state(2, index) - radars[0].initial_state(2),
                    targets[m].state(0, index) - radars[0].initial_state(0)
                );

                // Add noise to range and theta values
                range += normal_range(gen);
                theta += normal_theta(gen);

                detections.push_back({range, theta});
            }
        }

        Z[n].detections = detections;
    }

    return Z;
}