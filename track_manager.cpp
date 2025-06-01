#include "track_manager.h"
#include <algorithm> // For std::max

void track_managing(std::vector<Track>& track_list_update, const Parameter& param) {
    if (!track_list_update.empty()) {
        for (auto& track : track_list_update) {
            // Track termination
            if (track.status != -1) { // If track is not already dead
                int sum = 0;
                size_t update_length = track.track_info.size();

                // Confirmed track termination
                if (update_length > param.track.n) {  // Corrected
                    for (size_t j = update_length - param.track.n; j < update_length; ++j) {  // Corrected
                        sum += track.track_info[j].is_associated;
                    }
                    if (sum < param.track.m2) {  // Corrected
                        track.status = -1; // Terminate
                        track.track_info.back().status = -1;
                    }
                }
                // Tentative track termination
                else if ((param.track.n - update_length) < param.track.m1) {  // Corrected
                    for (size_t j = 0; j < update_length; ++j) {
                        sum += track.track_info[j].is_associated;
                    }
                    if (sum < (param.track.m1 - (param.track.n - update_length))) {  // Corrected
                        track.status = -1; // Terminate
                        track.track_info.back().status = -1;
                    }
                }
                else if (param.track.n == update_length) {  // Corrected
                    for (size_t j = 0; j < update_length; ++j) {
                        sum += track.track_info[j].is_associated;
                    }
                    if (sum < param.track.m1) {  // Corrected
                        track.status = -1; // Terminate
                        track.track_info.back().status = -1;
                    }
                }
            }

            // Track promotion
            if (track.status == 0) { // If tentative track
                int sum = 0;
                size_t update_length = track.track_info.size();

                for (size_t j = std::max(update_length - param.track.n, size_t(0)); j < update_length; ++j) {  // Corrected
                    sum += track.track_info[j].is_associated;
                }
                if (sum >= param.track.m1) {  // Corrected
                    track.status = 1; // Promote or confirm
                    track.track_info.back().status = 1;
                }
            }
        }
    }
} 