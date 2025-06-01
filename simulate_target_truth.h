#ifndef SIMULATE_TARGET_TRUTH_H
#define SIMULATE_TARGET_TRUTH_H

#include "parameters.h"
#include <vector>
#include <Eigen/Dense>

// No need to return TargetTruth anymore
void Simulate_target_truth(const Parameter& param, Target& target, int option, int scenario);

#endif // SIMULATE_TARGET_TRUTH_H
