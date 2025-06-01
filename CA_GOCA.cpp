#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <utility>
#include <string>
#include <Eigen/Eigenvalues>
#include <vector>
#include <complex>
#include <Eigen/Dense>
#include "readDCA1000.h"
#include<unsupported/Eigen/CXX11/Tensor>
#include<unsupported/Eigen/FFT>
#include <sciplot/sciplot.hpp>
#include <iomanip> 
#include <chrono>
#include <thread>
#include <cstdio>


void applyCFAR(const Eigen::MatrixXcd& RangeDopM, 
    const std::vector<int>& dop_grid, 
    const std::vector<int>& rng_grid,
    int frameNumber);
void displayCFAROutput(int rng_bins, int dop_bins, const Eigen::MatrixXd& abs_cfar_output, int frameNumber);


//Applying matched filter

    Eigen::MatrixXcd applyMatchedFiltering(const Eigen::MatrixXcd& data, const Eigen::RowVectorXcd& reference_chirp) {
    int nS = data.rows();
    int odd_chirps = data.cols();
    
    // Initialize output matrix with zeros
    Eigen::MatrixXcd output_signal_full = Eigen::MatrixXcd::Zero(nS, odd_chirps);
    
    // Create matched filter (time-reversed conjugate of reference chirp)
    Eigen::VectorXcd matched_filter(reference_chirp.size());
    for (int i = 0; i < reference_chirp.size(); ++i) {
        // Reverse and conjugate (flipud(conj())
        matched_filter(i) = std::conj(reference_chirp(reference_chirp.size() - 1 - i));
    }
    
    // Apply matched filtering for each pulse
    for (int pulse = 0; pulse < odd_chirps; ++pulse) {
        // Extract current pulse as column vector
        Eigen::VectorXcd received_signal = data.col(pulse);
        
        
        // This returns the central part of the convolution with the same size as received_signal
        for (int i = 0; i < nS; ++i) {
            std::complex<double> sum(0, 0);
            for (int j = 0; j < matched_filter.size(); ++j) {
                int idx = i - j + matched_filter.size() / 2;
                if (idx >= 0 && idx < nS) {
                    sum += received_signal(idx) * matched_filter(j);
                }
            }
            output_signal_full(i, pulse) = sum;
        }
    }
    
    return output_signal_full;
}


// Function declaration
using namespace Eigen;
using Complex = std::complex<double>;

// Function declaration
MatrixXcd helperMTIFilter(const MatrixXcd& x, double cdn = 0.0);

MatrixXcd helperMTIFilter(const MatrixXcd& x, double cdn) {
    // Filter coefficients (equivalent to h = [1 -1])
    VectorXcd h(2);
    h << Complex(1.0, 0.0), Complex(-1.0, 0.0);
    
    // Alternative filter coefficients (commented out)
    // VectorXcd h(3); h << 1.0, -2.0, 1.0; // 3-tap
    // VectorXcd h(4); h << 1.0, -3.0, 3.0, -1.0; // 4-tap
    
    // If non-zero center Doppler, apply phase ramp to filter coefficients
    if (cdn != 0.0) {
        for (int i = 0; i < h.size(); ++i) {
            double phi = 2.0 * M_PI * cdn * static_cast<double>(i);
            h(i) = h(i) * Complex(std::cos(phi), std::sin(phi));
        }
    }
    
    // Get input dimensions
    int rows = x.rows();
    int cols = x.cols();
    
    // Initialize output matrix
    MatrixXcd y(rows, cols - h.size() + 1);
    y.setZero();
    
    // Run filter along columns
    for (int i = 0; i < rows; ++i) {
        for (int j = h.size() - 1; j < cols; ++j) {
            Complex sum(0.0, 0.0);
            for (int k = 0; k < h.size(); ++k) {
                sum += h(k) * x(i, j - k);
            }
            y(i, j - h.size() + 1) = sum;
        }
    }
    
    // Calculate noise gain
    double noiseGain = h.squaredNorm();
    noiseGain = std::sqrt(noiseGain);
    
    // Normalize output
    y /= noiseGain;
    
    return y;
}

Eigen::MatrixXcd fftshift(const Eigen::MatrixXcd& mat) {
    int rows = mat.rows(), cols = mat.cols();
    int row_shift = rows / 2, col_shift = cols / 2;

    Eigen::MatrixXcd shifted(rows, cols);
    shifted.topLeftCorner(row_shift, col_shift) = mat.bottomRightCorner(row_shift, col_shift);
    shifted.topRightCorner(row_shift, col_shift) = mat.bottomLeftCorner(row_shift, col_shift);
    shifted.bottomLeftCorner(row_shift, col_shift) = mat.topRightCorner(row_shift, col_shift);
    shifted.bottomRightCorner(row_shift, col_shift) = mat.topLeftCorner(row_shift, col_shift);

    return shifted;
}


// Gnuplot pipe declaration as a global variable
FILE* gnuplotPipe = nullptr;
FILE* gnuplotPipeCFAR = nullptr;



// Function implementations
void initGnuplot() {
    // Open Gnuplot pipe if not already open
    if (!gnuplotPipe) {
        #ifdef _WIN32
            // Windows-specific Gnuplot pipe opening
            gnuplotPipe = _popen("gnuplot", "w");
        #else
            // Unix/Linux-specific Gnuplot pipe opening
            gnuplotPipe = popen("gnuplot", "w");
        #endif

        if (!gnuplotPipe) {
            std::cerr << "Error: Could not open Gnuplot pipe." << std::endl;
            return;
        }
    }

    if (!gnuplotPipeCFAR) {
        #ifdef _WIN32
            gnuplotPipeCFAR = _popen("gnuplot", "w");
        #else
            gnuplotPipeCFAR = popen("gnuplot", "w");
        #endif
        if (!gnuplotPipeCFAR) {
            std::cerr << "Error: Could not open CFAR Gnuplot pipe." << std::endl;
        }
    }
}



void displayRDMap(int nS, const Eigen::MatrixXd& abs_final_fft, int frameNumber) {
    // Initialize Gnuplot if not already done
    initGnuplot();

    // Check if the Gnuplot pipe is available
    if (!gnuplotPipe) {
        std::cerr << "Error: Gnuplot pipe not available." << std::endl;
        return;
    }

    // Check if the data is empty
    if (abs_final_fft.rows() == 0 || abs_final_fft.cols() == 0) {
        std::cerr << "Error: fft is empty." << std::endl;
        return;
    }
    
    // Create a temporary data file for fft data
    std::ofstream dataFile("temp_fft_data.txt");
    if (!dataFile.is_open()) {
        std::cerr << "Error: Could not open temporary file for data." << std::endl;
        return;
    }
    
    // Determine plot type based on data dimensions
    bool is2D = (abs_final_fft.rows() > 1 && abs_final_fft.cols() > 1);
    
    // Write data to file
    if (!is2D) {
        // 1D plot (either row or column vector)
        for (int i = 0; i < std::max(abs_final_fft.rows(), abs_final_fft.cols()); ++i) {
            double value = (abs_final_fft.rows() == 1) ? 
                std::abs(abs_final_fft(0, i)) : 
                std::abs(abs_final_fft(i, 0));
            dataFile << i << " " << value << std::endl;
        }
    } else {
        //
        for (int i = 0; i < abs_final_fft.rows(); ++i) {
            for (int j = 0; j < abs_final_fft.cols(); ++j) {
                dataFile << i << " " << j << " " << std::abs(abs_final_fft(i, j)) << std::endl;
            }
            dataFile << std::endl; // Empty line for gnuplot
        }
    }
    dataFile.close();

    // Send GNUPlot commands
    fprintf(gnuplotPipe, "set terminal qt size 800,600 title 'Range doppler'\n");
    fprintf(gnuplotPipe, "set pm3d map\n");
    fprintf(gnuplotPipe, "set palette defined (0 'blue', 1 'cyan', 2 'yellow', 3 'red')\n");
    fprintf(gnuplotPipe, "set xlabel 'Doppler Bins'\n");
    fprintf(gnuplotPipe, "set ylabel 'Range Bins'\n");
    fprintf(gnuplotPipe, "set yrange [250:0]\n");
    fprintf(gnuplotPipe, "set title 'RD Map - Frame %d'\n", frameNumber);
    
    fprintf(gnuplotPipe, "splot '-' using 1:2:3 with image\n");

    // Send the data directly through the pipe
    for (int i = 0; i < nS; ++i) {
        for (int j = 0; j < abs_final_fft.cols(); ++j) {
            fprintf(gnuplotPipe, "%d %d %f\n", j, i, abs_final_fft(i, j));
        }
        fprintf(gnuplotPipe, "\n");
    }
    
    fprintf(gnuplotPipe, "e\n");
    fflush(gnuplotPipe);
    
    // Display and then close
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::cout << "RD Map Frame " << frameNumber << " displayed " << std::endl;
}


// Replace 0 and NaN values in a 2D matrix with the mean of valid elements
void fixZeroOrNaNValues(Eigen::MatrixXd& RangeDopM) {
    // Replace zeros with mean of non-zero elements
    double meanVal = RangeDopM.array().select(0, RangeDopM).sum() / (RangeDopM.array() > 0).count();
    RangeDopM = RangeDopM.unaryExpr([meanVal](double x) {
        return (x == 0.0) ? meanVal : x;
    });

    // Replace NaNs with mean of non-NaN elements
    double sum = 0.0;
    int count = 0;
    for (int i = 0; i < RangeDopM.rows(); ++i) {
        for (int j = 0; j < RangeDopM.cols(); ++j) {
            if (!std::isnan(RangeDopM(i, j))) {
                sum += RangeDopM(i, j);
                count++;
            }
        }
    }
    double nanMean = (count > 0) ? sum / count : 0.0;
    for (int i = 0; i < RangeDopM.rows(); ++i) {
        for (int j = 0; j < RangeDopM.cols(); ++j) {
            if (std::isnan(RangeDopM(i, j))) {
                RangeDopM(i, j) = nanMean;
            }
        }
    }
}

// Generate axis values (1,2,3,...,n)
std::vector<int> generateAxis(int n) {
    std::vector<int> axis(n);
    std::iota(axis.begin(), axis.end(), 1); // Fill with 1,2,3,...,n
    return axis;
}


//  2D CA-CFAR Implementation (Range-Doppler)
void applyCFAR(const Eigen::MatrixXcd& RangeDopM, 
    const std::vector<int>& dop_grid, 
    const std::vector<int>& rng_grid,
    int frameNumber) {
// CFAR parameters - can be moved to function parameters for flexibility
int Tr = 16;         // Training cells in range dimension
int Td = 16;          // Training cells in doppler dimension
int Gr = 2;          // Guard cells in range dimension
int Gd = 2;          // Guard cells in doppler dimension
double offset_db = 14.0;  // Detection threshold offset in dB
double PFA = 1e-6;        // Probability of false alarm (alternative to fixed offset)

int rng_bins = RangeDopM.rows();
int dop_bins = RangeDopM.cols();

// Initialize CFAR output and detected targets matrices
Eigen::MatrixXcd cfar_output = Eigen::MatrixXcd::Zero(rng_bins, dop_bins);
Eigen::MatrixXi detected_targets = Eigen::MatrixXi::Zero(rng_bins, dop_bins);

// Convert to power domain for proper CFAR processing
Eigen::MatrixXd power_data = RangeDopM.cwiseAbs().array().square();

/* Compute alpha based on PFA and number of training cells
 For CA-CFAR: alpha = (pow(PFA, -1/(N)) - 1)*/
int N = 2 * (Tr + Td + Tr * Td) - 2 * (Gr + Gd + Gr * Gd);  // Number of training cells
double alpha = std::pow(PFA, -1.0/N) - 1.0;
//double alpha = offset_db;

// Use either fixed offset or PFA-based alpha
bool use_pfa = false;  // Set to true to use PFA instead of fixed offset
double threshold_factor = use_pfa ? alpha : std::pow(10.0, offset_db / 10.0);
//double threshold_factor=alpha;

// Process each cell in the range-doppler map
for (int d = Td + Gd; d < dop_bins - (Td + Gd); ++d) {
for (int r = Tr + Gr; r < rng_bins - (Tr + Gr); ++r) {
 // Extract training cells (excluding guard cells) in 2D window
 std::vector<double> training_cells;
 training_cells.reserve(2 * (Tr * Td)); // Approximately reserve space

 // Collect training cells in 2D window
 for (int i = d - (Td + Gd); i <= d + (Td + Gd); ++i) {
     for (int j = r - (Tr + Gr); j <= r + (Tr + Gr); ++j) {
         // Skip guard cells and CUT (Cell Under Test)
         if ((std::abs(i - d) <= Gd && std::abs(j - r) <= Gr) || (i == d && j == r)) {
             continue;
         }

         // Ensure we stay within bounds
         if (i >= 0 && i < dop_bins && j >= 0 && j < rng_bins) {
             training_cells.push_back(power_data(j, i));
         }
     }
 }

 // Compute the threshold using mean or median
 double noise_level = 0.0;
 if (training_cells.size() > 0) {
     // Option 1: Mean estimator (CA-CFAR)
     for (double val : training_cells) {
         noise_level += val;
     }
     noise_level /= training_cells.size();
     
     

     // Calculate threshold
     double threshold = noise_level * threshold_factor;
     
     // Apply CFAR detection
     if (power_data(r, d) > threshold) {
         cfar_output(r, d) = RangeDopM(r, d);  // Keep original complex value
         detected_targets(r, d) = 1;           // Mark detection
     }
 }
}
}

// Calculate absolute values for display
Eigen::MatrixXd abs_cfar_output = cfar_output.cwiseAbs();

// Apply non-maximum suppression to reduce multiple detections of the same target
Eigen::MatrixXd suppressed_output = Eigen::MatrixXd::Zero(rng_bins, dop_bins);
for (int d = 1; d < dop_bins - 1; ++d) {
for (int r = 1; r < rng_bins - 1; ++r) {
 if (detected_targets(r, d) == 1) {
     // Check if this is a local maximum in a 3x3 window
     bool is_local_max = true;
     double center_val = abs_cfar_output(r, d);
     
     for (int i = -1; i <= 1; ++i) {
         for (int j = -1; j <= 1; ++j) {
             if (i == 0 && j == 0) continue;
             
             if (abs_cfar_output(r+i, d+j) > center_val) {
                 is_local_max = false;
                 break;
             }
         }
         if (!is_local_max) break;
     }
     
     if (is_local_max) {
         suppressed_output(r, d) = abs_cfar_output(r, d);
     }
 }
}
}

// Display the CFAR output using GNUPlot
displayCFAROutput(rng_bins, dop_bins, suppressed_output, frameNumber);

/* Optionally, save detected targets to a file
if (frameNumber % 10 == 0) {  // Save every 10th frame
std::ofstream outFile("cfar_detections_frame_" + std::to_string(frameNumber) + ".txt");
if (outFile.is_open()) {
 for (int d = 0; d < dop_bins; ++d) {
     for (int r = 0; r < rng_bins; ++r) {
         if (suppressed_output(r, d) > 0) {
             // Save range bin, doppler bin, and amplitude
             outFile << r << " " << d << " " << suppressed_output(r, d) << std::endl;
         }
     }
 }
 outFile.close();
}
}*/

}

// Function to display the CFAR output with visualization
void displayCFAROutput(int rng_bins, int dop_bins, const Eigen::MatrixXd& abs_cfar_output, int frameNumber) {
    // Initialize Gnuplot if not already done
    initGnuplot();

    // Check if the Gnuplot pipe is available
    if (!gnuplotPipeCFAR) {
        std::cerr << "Error: CFAR Gnuplot pipe not available." << std::endl;
        return;
    }

// Normalize the output for better visualization
double max_val = abs_cfar_output.maxCoeff();
if (max_val <= 0) max_val = 1.0;  // Prevent division by zero

Eigen::MatrixXd normalized_output = abs_cfar_output / max_val;

// Send GNUPlot commands
fprintf(gnuplotPipeCFAR, "set terminal qt size 800,600 title 'CFAR Detections'\n");
fprintf(gnuplotPipeCFAR, "set pm3d map\n");
fprintf(gnuplotPipeCFAR, "set palette defined (0 'blue', 1 'cyan', 2 'yellow', 3 'red')\n");
//fprintf(gnuplotPipeCFAR, "set palette defined (0 '#0000FF', 0.25 '#00FFFF', 0.5 '#90EE90', 0.75 '#FFA500', 1 '#FFFF00')\n");
fprintf(gnuplotPipeCFAR, "set xlabel 'Doppler Bins'\n");
fprintf(gnuplotPipeCFAR, "set ylabel 'Range Bins'\n");
fprintf(gnuplotPipeCFAR, "set yrange [%d:0]\n", rng_bins);
fprintf(gnuplotPipeCFAR, "set xrange [0:%d]\n", dop_bins);
fprintf(gnuplotPipeCFAR, "set title 'CA-CFAR Output - Frame %d'\n", frameNumber);
fprintf(gnuplotPipeCFAR, "set colorbar\n");
fprintf(gnuplotPipeCFAR, "set grid\n");

// Optional: Add a timestamp to the plot
time_t now = time(nullptr);
char timestamp[64];
strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
fprintf(gnuplotPipeCFAR, "set label 'Time: %s' at graph 0.02, 0.95\n", timestamp);

// Plot data
fprintf(gnuplotPipeCFAR, "splot '-' using 1:2:3 with image\n");

// Send the data directly through the pipe
for (int i = 0; i < rng_bins; ++i) {
for (int j = 0; j < dop_bins; ++j) {
 if (normalized_output(i, j) > 0) {  // Only send non-zero values
     fprintf(gnuplotPipeCFAR, "%d %d %f\n", j, i, normalized_output(i, j));
 } else {
     fprintf(gnuplotPipeCFAR, "%d %d 0\n", j, i);
 }
}
fprintf(gnuplotPipeCFAR, "\n");
}

fprintf(gnuplotPipeCFAR, "e\n");
fflush(gnuplotPipeCFAR);

// Display and then wait briefly
std::this_thread::sleep_for(std::chrono::milliseconds(50));
std::cout << "CA-CFAR Output Frame " << frameNumber << " displayed with " 
   << (normalized_output.array() > 0).count() << " detections" << std::endl;
}




// GOCA-CFAR Implementation (Range-Doppler)
void applyGOCACFAR(const Eigen::MatrixXcd& RangeDopM, 
    const std::vector<int>& dop_grid, 
    const std::vector<int>& rng_grid,
    int frameNumber) {
// GOCA-CFAR parameters 
int trc_num = 8;      // Training cells
int guac_num = 2;      // Guard cells
double offset = 1.3;             // Detection threshold offset

int rng_bins = RangeDopM.rows();
int dop_bins = RangeDopM.cols();

// Initialize CFAR output and detected targets matrices
Eigen::MatrixXcd cfar_output = Eigen::MatrixXcd::Zero(rng_bins, dop_bins);
Eigen::MatrixXi detected_targets = Eigen::MatrixXi::Zero(rng_bins, dop_bins);

// Convert to power domain for proper CFAR processing
Eigen::MatrixXd power_data = RangeDopM.cwiseAbs().array().square();

// Process each cell in the range-doppler map
for (int r = 0; r < rng_bins; ++r) {
for (int c = 0; c < dop_bins; ++c) {
// Define boundaries for guard and training cells
int min_y_guard = std::max(0, r - guac_num);
int max_y_guard = std::min(rng_bins - 1, r + guac_num);
int min_x_guard = std::max(0, c - guac_num);
int max_x_guard = std::min(dop_bins - 1, c + guac_num);

int min_y_train = std::max(0, r - (guac_num + trc_num));
int max_y_train = std::min(rng_bins - 1, r + (guac_num + trc_num));
int min_x_train = std::max(0, c - (guac_num + trc_num));
int max_x_train = std::min(dop_bins - 1, c + (guac_num + trc_num));

// Create guard and training cell matrices
Eigen::MatrixXd guard_matrix = Eigen::MatrixXd::Zero(max_y_guard - min_y_guard + 1, max_x_guard - min_x_guard + 1);
Eigen::MatrixXd train_matrix = Eigen::MatrixXd::Zero(max_y_train - min_y_train + 1, max_x_train - min_x_train + 1);

// Fill guard matrix
for (int i = min_y_guard; i <= max_y_guard; ++i) {
  for (int j = min_x_guard; j <= max_x_guard; ++j) {
      guard_matrix(i - min_y_guard, j - min_x_guard) = power_data(i, j);
  }
}

// Fill training matrix
for (int i = min_y_train; i <= max_y_train; ++i) {
  for (int j = min_x_train; j <= max_x_train; ++j) {
      train_matrix(i - min_y_train, j - min_x_train) = power_data(i, j);
  }
}

// Subtract guard cells from training cells
// Map guard cells to corresponding positions in the training matrix
int guard_offset_y = guac_num;
int guard_offset_x = guac_num;

// Relative position of CUT in the training matrix
int r_train = r - min_y_train;
int c_train = c - min_x_train;

// Create a modified data matrix (equivalent to newdatamatrix in MATLAB)
Eigen::MatrixXd data_matrix = train_matrix;

// Set guard cells to 0 (including CUT)
for (int i = 0; i < guard_matrix.rows(); ++i) {
  for (int j = 0; j < guard_matrix.cols(); ++j) {
      int train_i = i + (r_train - guac_num);
      int train_j = j + (c_train - guac_num);
      
      // Check if within bounds of data_matrix
      if (train_i >= 0 && train_i < data_matrix.rows() && 
          train_j >= 0 && train_j < data_matrix.cols()) {
          data_matrix(train_i, train_j) = 0;
      }
  }
}

// Mark CUT position with 1 (for identification)
data_matrix(r_train, c_train) = 1;

// Apply GOCA-CFAR technique
double threshold_val = 0.0;

// Check if CUT is not at the edge
if (c_train > 0 && c_train < data_matrix.cols() - 1) {
  // Split data into lead and lag windows
  Eigen::MatrixXd lead = data_matrix.block(0, 0, data_matrix.rows(), c_train);
  Eigen::MatrixXd lag = data_matrix.block(0, c_train + 1, data_matrix.rows(), data_matrix.cols() - c_train - 1);
  
  int N_lead = lead.rows() * lead.cols();
  int N_lag = lag.rows() * lag.cols();
  
  // Extract column containing CUT and remove the 1 (CUT marker)
  std::vector<double> near;
  for (int i = 0; i < data_matrix.rows(); ++i) {
      if (data_matrix(i, c_train) != 1) {
          near.push_back(data_matrix(i, c_train));
      }
  }
  
  // Split near cells into two parts (a and b)
  std::vector<double> a, b;
  if (near.size() % 2 == 0) {
      a.assign(near.begin(), near.begin() + near.size() / 2);
      b.assign(near.begin() + near.size() / 2, near.end());
  } else {
      a.assign(near.begin(), near.begin() + (near.size() + 1) / 2);
      b.assign(near.begin() + (near.size() + 1) / 2, near.end());
  }
  
  int N_a = a.size();
  int N_b = b.size();
  
  // Calculate lead and lag averages
  double lead_sum = lead.sum();
  double a_sum = 0;
  for (double val : a) a_sum += val;
  
  double lag_sum = lag.sum();
  double b_sum = 0;
  for (double val : b) b_sum += val;
  
  // Average values
  double lead_avg = (lead_sum + a_sum) / (N_lead + N_a);
  double lag_avg = (lag_sum + b_sum) / (N_lag + N_b);
  
  // Use maximum of lead and lag for threshold (GOCA)
  //double value = std::max(lead_avg, lag_avg);

  // Use minimum of lead and lag for threshold (SOCA)
double value = std::min(lead_avg, lag_avg);
  
  // Convert to dB and apply offset
  threshold_val = 10 * std::log10(value) * offset;
} else {
  // If CUT is at edge, use sum of all training cells
  double sum_total = data_matrix.sum() - 1.0; // Subtract the 1 used as marker
  double avg = sum_total / (data_matrix.size() - 1); // Average excluding the CUT
  
  // Convert to dB and apply offset
  threshold_val = 10 * std::log10(avg) * offset;
}

// Apply detection
double cut_val = 10 * std::log10(power_data(r, c));
if (cut_val >= threshold_val) {
  cfar_output(r, c) = RangeDopM(r, c);
  detected_targets(r, c) = 1;
}
}
}

// Calculate absolute values for display
Eigen::MatrixXd abs_cfar_output = cfar_output.cwiseAbs();

// Apply non-maximum suppression to reduce multiple detections of the same target
Eigen::MatrixXd suppressed_output = Eigen::MatrixXd::Zero(rng_bins, dop_bins);
for (int d = 1; d < dop_bins - 1; ++d) {
for (int r = 1; r < rng_bins - 1; ++r) {
if (detected_targets(r, d) == 1) {
  // Check if this is a local maximum in a 3x3 window
  bool is_local_max = true;
  double center_val = abs_cfar_output(r, d);
  
  for (int i = -1; i <= 1; ++i) {
      for (int j = -1; j <= 1; ++j) {
          if (i == 0 && j == 0) continue;
          
          if (abs_cfar_output(r+i, d+j) > center_val) {
              is_local_max = false;
              break;
          }
      }
      if (!is_local_max) break;
  }
  
  if (is_local_max) {
      suppressed_output(r, d) = abs_cfar_output(r, d);
  }
}
}
}

// Display the CFAR output using GNUPlot
displayCFAROutput(rng_bins, dop_bins, suppressed_output, frameNumber);
}

int main() {
    std::string filename = "C:\\Users\\SWEEKRUTHI\\sashaprayathi\\sr_car.bin";
    Eigen::MatrixXcd retVal = readDCA1000(filename);

    if (retVal.size() == 0) {
        std::cerr << "Error: Unable to read data from file." << std::endl;
        return -1;
    }

    int frames = 100;
    int s = static_cast<int>(retVal.cols()) / frames;
    int nS = 256;  // Samples per chirp
    int m_chirps = s / nS;  // Total chirps per frame
    int ant_ele = retVal.rows();  // Number of receiving antennas
    int arr_ele = 8;
    int K=256;
    double d=0.5;
    int noise_var=1;
    int r=1;
    int t=0;
    int ADC_Samples = 256;
    const double SampleRate = 5000e3; // 5000e3 is equivalent to 5,000,000.0
    const double Freq_Slope = 8.0144e12;

    

    // Initialize tensor for chirp data
    Eigen::Tensor<std::complex<double>, 3> data_chirp(ant_ele, nS, m_chirps);
    data_chirp.setZero();  // Avoid uninitialized memory issues

    // Extract frame and process chirps
    for (int f = 0; f < frames; ++f) {
        Eigen::MatrixXcd data_frame = retVal.middleCols(f * s, s);

        // Extract chirps from the frame
        for (int nC = 0; nC < m_chirps; ++nC) {
            if (nC * nS + nS <= data_frame.cols()) {  // Prevent out-of-bounds
                Eigen::MatrixXcd temp_data = data_frame.middleCols(nC * nS, nS);
                for (int i = 0; i < ant_ele; ++i) {
                    for (int j = 0; j < nS; ++j) {
                        data_chirp(i, j, nC) = temp_data(i, j);
                    }
                }
            }
        }
    

    // Extract only odd-indexed chirps -   1, 3, 5, .
    int odd_chirps = (m_chirps + 1) / 2;
    Eigen::Tensor<std::complex<double>, 3> chirp_odd(nS, ant_ele, odd_chirps);  

    int odd_index = 0;
    for (int k = 0; k < m_chirps; k += 2) {  
        for (int j = 0; j < ant_ele; ++j) {  // Rx first (columns
            for (int i = 0; i < nS; ++i) {  // Samples second (rows)
                chirp_odd(i, j, odd_index) = data_chirp(j, i, k);  // Fix indexing order
            }
        }
        odd_index++;
    }

      Eigen::Tensor<std::complex<double>, 3> chirp_odd_one(nS, 1, odd_chirps);
    for (int nC = 0; nC < odd_chirps; ++nC) {
        for (int i = 0; i < nS; ++i) {
            chirp_odd_one(i, 0, nC) = chirp_odd(i, 0, nC);  // Get first antenna (index 0)
        }
    }
    
    // Squeeze the data (convert 3D tensor to 2D matrix)
    Eigen::MatrixXcd data(nS, odd_chirps);
    for (int nC = 0; nC < odd_chirps; ++nC) {
        for (int i = 0; i < nS; ++i) {
            data(i, nC) = chirp_odd_one(i, 0, nC);
        }
    }
    
    

// Generate time vector
std::vector<double> timeVec(ADC_Samples);
for (int i = 0; i < ADC_Samples; ++i) {
    timeVec[i] = static_cast<double>(i) / SampleRate;
}

// Generate reference chirp
Eigen::RowVectorXcd reference_chirp(ADC_Samples);
for (int i = 0; i < ADC_Samples; ++i) {
    double phase = M_PI * Freq_Slope * timeVec[i] * timeVec[i];
    reference_chirp(i) = std::exp(std::complex<double>(0, phase));
}
  

// Apply matched filtering
Eigen::MatrixXcd output_signal_full = applyMatchedFiltering(data, reference_chirp);
    
auto MTI_data = helperMTIFilter(output_signal_full);

 // Apply FFT along rows (range FFT)
 Eigen::FFT<double> fft;
 Eigen::MatrixXcd range_fft(nS, MTI_data.cols());
 for (int i = 0; i < nS; ++i) {
     Eigen::VectorXcd row = MTI_data.row(i);
     Eigen::VectorXcd fft_result(nS);
     fft.fwd(fft_result, row);
     range_fft.row(i) = fft_result;
 }

 std::cout << "Range FFT Dimensions: " << range_fft.rows() << " x " << range_fft.cols() << std::endl;

 // Transpose 256x64 -> 64x256
 Eigen::MatrixXcd transposed_range_fft = range_fft.transpose();

 // Apply FFT along rows (Azimuth FFT)
 Eigen::MatrixXcd azimuth_fft(MTI_data.cols(), nS);
 for (int i = 0; i < MTI_data.cols(); ++i) {
     Eigen::VectorXcd row = transposed_range_fft.row(i);
     Eigen::VectorXcd fft_result(nS);
     fft.fwd(fft_result, row);
     azimuth_fft.row(i) = fft_result;
 }

 // Transpose back (256x64)
 Eigen::MatrixXcd final_fft = azimuth_fft.transpose();


 final_fft = fftshift(final_fft);

 // absolute value
 Eigen::MatrixXd abs_final_fft = final_fft.cwiseAbs();
 //std::cout << "Range-Doppler FFT Dimensions: " << abs_final_fft.rows() << " x " << abs_final_fft.cols() << std::endl;

 displayRDMap(nS, abs_final_fft,f);

 
 //cfar
Eigen::MatrixXd RangeDopM = abs_final_fft;  // Use absolute value matrix
fixZeroOrNaNValues(RangeDopM);

// Get the dimensions
int dop_bins = RangeDopM.cols();
int rng_bins = RangeDopM.rows();

//std::cout<<"Doppler Bins: "<<dop_bins<<std::endl;
//std::cout<<"Range Bins: "<<rng_bins<<std::endl;

// Generate axis vectors
std::vector<int> dop_grid = generateAxis(dop_bins);
std::vector<int> rng_grid = generateAxis(rng_bins); 

// After setting up RangeDopM, dop_grid, rng_grid
//applyCFAR(RangeDopM, dop_grid, rng_grid, f);

applyGOCACFAR(RangeDopM, dop_grid, rng_grid, f);


}
    return 0;
}