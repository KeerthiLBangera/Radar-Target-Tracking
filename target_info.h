#ifndef TARGET_INFO_H
#define TARGET_INFO_H

#include "parameters.h"  // ✅ Use the Target struct defined in parameters.h

// ✅ Updated declaration to accept parameters for simulation
std::vector<Target> target_info(const Parameter& param);

#endif // TARGET_INFO_H
