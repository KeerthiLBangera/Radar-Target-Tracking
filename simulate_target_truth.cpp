#include "Simulate_target_truth.h"
#include <cmath>
#include <algorithm>

void Simulate_target_truth(const Parameter& param, Target& target, int option, int scenario) {
    // Align start and end times with radar sampling
    double time_start = param.radar.k_start + param.radar.T * std::ceil((target.start - param.radar.k_start) / param.radar.T);
    double time_end   = param.radar.k_start + param.radar.T * std::floor((target.end - param.radar.k_start) / param.radar.T);

    // Generate time vector
    std::vector<double> time_vec;
    for (double t = time_start; t <= time_end; t += param.radar.T)
        time_vec.push_back(t);

    target.time = time_vec;
    int N = time_vec.size();
    target.state = Eigen::MatrixXd(4, N);

    // Set initial state
    Eigen::Vector4d init_state = target.initial_state;
    if (init_state(1) == 0 && init_state(3) == 0) {
        init_state(1) = 10.0;  // default vx
        init_state(3) = 10.0;  // default vy
    }

    target.state.col(0) = init_state;

    for (int i = 1; i < N; ++i) {
        Eigen::Matrix4d F;

        // Choose motion model
        if (scenario == 1 && std::find(target.turn.begin(), target.turn.end(), static_cast<int>(time_vec[i])) != target.turn.end()) {
            F = param.F_2(target.turn_rate, param.radar.T);
        } else if (scenario == 2 && static_cast<int>(time_vec[i]) == 0) {
            F = param.F_2(param.filter.W_min, param.radar.T);
        } else {
            F = param.F_1(param.radar.T);
        }

        // Update state without process noise (for clean paths)
        target.state.col(i) = F * target.state.col(i - 1);
    }
}
