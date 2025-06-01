#include <iostream>
#include <fstream>
#include <cmath>
#include <complex>
#include <vector>
#include <chrono>
#include <thread>
#include <cstdio>
#include <iomanip>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "readDCA1000.h"

constexpr double c = 299792458;               // Speed of light (m/s)
constexpr double SampleRate = 5000e3;          // Sample rate (5e6 samples/s)
constexpr int ADC_Samples = 256;               // ADC samples per chirp

constexpr double Freq_Slope = 8e12;       // Chirp slope (Hz/s)
constexpr double idletime = 3e-6;
constexpr double rampendtime = 56e-6;
constexpr double carrier_frequency = 77e9;     // 77 GHz
constexpr double start_freq = carrier_frequency; // Starting frequency
double theta=0;
double rho=0;

struct RangeAnglePair {
    double range;
    double angle;
};

struct NoisePoint {
    double angle;
    double range;
};

struct FrameData {
    std::vector<Eigen::MatrixXcd> frames;
    std::vector<std::pair<double, double>> mean1;  // {angle, range}
    std::vector<std::pair<double, double>> mean2;  // {angle, range}
    std::vector<bool> flag1;
    std::vector<bool> flag2;
    
    // Fields for detection tracking
    std::vector<int> detections;
    std::vector<std::vector<RangeAnglePair>> range_angle_pairs;
    std::vector<std::vector<NoisePoint>> noise;
    std::vector<double> time;
};

// Compute range resolution
double getRangeResolution() {
    return (c * SampleRate) / (2.0 * Freq_Slope * ADC_Samples);
}

// Create a time vector (seconds)
std::vector<double> createTimeVector(int N, double sampleRate) {
    std::vector<double> t(N);
    for (int i = 0; i < N; ++i) {
        t[i] = static_cast<double>(i) / sampleRate;
    }
    return t;
}

// Custom Cooley-Tukey FFT (in-place, recursive, assumes power-of-2 size)

void custom_fft(Eigen::VectorXcd& data) {
    const double PI = 3.14159265358979323846;
    int N = data.size();
    if (N <= 1) return;

    // Create new vectors for even and odd instead of resizing
    Eigen::VectorXcd even = Eigen::VectorXcd::Zero(N / 2);
    Eigen::VectorXcd odd = Eigen::VectorXcd::Zero(N / 2);
    
    // Split into even and odd indices
    for (int i = 0; i < N / 2; ++i) {
        even(i) = data(2 * i);
        odd(i) = data(2 * i + 1);
    }

    // Recursive FFT on even and odd parts
    custom_fft(even);
    custom_fft(odd);

    // Combine results without resizing
    for (int k = 0; k < N / 2; ++k) {
        std::complex<double> t = std::polar(1.0, -2.0 * PI * k / N) * odd(k);
        data(k) = even(k) + t;
        data(k + N / 2) = even(k) - t;
    }
}

// fft_wrapper function 
Eigen::VectorXcd fft_wrapper(const Eigen::VectorXcd& input) {
   
    Eigen::VectorXcd result(input.size());
    for (int i = 0; i < input.size(); ++i) {
        result(i) = input(i);
    }
    custom_fft(result);
    return result;
}

// fftshift function 
Eigen::VectorXcd fftshift(const Eigen::VectorXcd& input) {
    int N = input.size();
    int half = N / 2;
    
    // Create a new vector for the output
    Eigen::VectorXcd output(N);
    
    // Swap first and second halves
    for (int i = 0; i < half; ++i) {
        output(i) = input(i + half);
    }
    
    for (int i = 0; i < N - half; ++i) {
        output(i + half) = input(i);
    }
    
    return output;
}
// Doppler FFT and virtual antenna combination
void perform_doppler_fft(
    const std::vector<Eigen::MatrixXcd>& Rangedata_odd,
    const std::vector<Eigen::MatrixXcd>& Rangedata_even,
    std::vector<Eigen::MatrixXcd>& DopData_odd,
    std::vector<Eigen::MatrixXcd>& DopData_even,
    std::vector<Eigen::MatrixXcd>& Angle_input,
    int Ne, int Nr, int Nd) {

    // Validate input dimensions
    if (Rangedata_odd.size() != Ne || Rangedata_even.size() != Ne) {
        std::cerr << "Error: Rangedata size mismatch. Expected " << Ne
                  << ", got odd: " << Rangedata_odd.size()
                  << ", even: " << Rangedata_even.size() << std::endl;
        return;
    }

    // Clear output vectors before adding new elements
    DopData_odd.clear();
    DopData_even.clear();
    
    // Initialize with correctly sized matrices
    for (int i = 0; i < Ne; i++) {
        DopData_odd.push_back(Eigen::MatrixXcd::Zero(Nr, Nd));
        DopData_even.push_back(Eigen::MatrixXcd::Zero(Nr, Nd));
    }

    // Perform Doppler FFT
    for (int i_2 = 0; i_2 < Ne; ++i_2) {
        for (int j_2 = 0; j_2 < Nr; ++j_2) {
            // Extract Doppler data for even chirps
            Eigen::VectorXcd doppler_data_even(Nd);
            for (int k = 0; k < Nd; ++k) {
                doppler_data_even(k) = Rangedata_even[i_2](j_2, k);
            }
            
            Eigen::VectorXcd fft_result_even = fftshift(fft_wrapper(doppler_data_even));
            
            // Copy result to output matrix element by element
            for (int k = 0; k < Nd; ++k) {
                DopData_even[i_2](j_2, k) = fft_result_even(k);
            }

            // Extract Doppler data for odd chirps
            Eigen::VectorXcd doppler_data_odd(Nd);
            for (int k = 0; k < Nd; ++k) {
                doppler_data_odd(k) = Rangedata_odd[i_2](j_2, k);
            }
            
            Eigen::VectorXcd fft_result_odd = fftshift(fft_wrapper(doppler_data_odd));
            
            // Copy result to output matrix element by element
            for (int k = 0; k < Nd; ++k) {
                DopData_odd[i_2](j_2, k) = fft_result_odd(k);
            }
        }
    }

    // Clear and properly initialize Angle_input
    Angle_input.clear();
    for (int i = 0; i < 2 * Ne; i++) {
        Angle_input.push_back(Eigen::MatrixXcd::Zero(Nr, Nd));
    }

    // Combine for virtual antennas - copy element by element
    for (int i = 0; i < Ne; ++i) {
        for (int r = 0; r < Nr; ++r) {
            for (int c = 0; c < Nd; ++c) {
                Angle_input[i](r, c) = DopData_odd[i](r, c);
                Angle_input[i + Ne](r, c) = DopData_even[i](r, c);
            }
        }
    }
}

// Doppler processing function
void process_doppler(
    const std::vector<Eigen::MatrixXcd>& Rangedata_odd,
    const std::vector<Eigen::MatrixXcd>& Rangedata_even,
    int Ne, int Nr, int Nd,
    std::vector<Eigen::MatrixXcd>& DopData_odd,
    std::vector<Eigen::MatrixXcd>& DopData_even,
    std::vector<Eigen::MatrixXcd>& Angle_input) {
    
    perform_doppler_fft(Rangedata_odd, Rangedata_even, DopData_odd, DopData_even, Angle_input, Ne, Nr, Nd);
}

// Angle FFT processing

void process_angle_fft(
    const std::vector<Eigen::MatrixXcd>& Angle_input,
    int Nr, int Nd, int fft_Ang,
    Eigen::Tensor<std::complex<double>, 3>& AngData) {
    
    // Taylor window for angle FFT
    Eigen::VectorXd taylorWindow = Eigen::VectorXd::Zero(fft_Ang);
    // Simple approximation of Taylor window (for demonstration)
    // In a real implementation, you would use a proper Taylor window function
    for (int i = 0; i < fft_Ang; ++i) {
        double x = static_cast<double>(i) / (fft_Ang - 1) * 2.0 - 1.0;
        taylorWindow(i) = 1.0 - 0.5 * std::abs(x);
    }
    
    // Resize the output tensor
    AngData.resize(Nr, fft_Ang, Nd);
    
    // Perform angle FFT
    for (int i_3 = 0; i_3 < Nd; ++i_3) {
        for (int j_3 = 0; j_3 < Nr; ++j_3) {
            // Extract data for this range-doppler bin across all antennas
            Eigen::VectorXcd win_xcube(fft_Ang);
            for (int ant = 0; ant < fft_Ang; ++ant) {
                if (ant < Angle_input.size()) {
                    win_xcube(ant) = Angle_input[ant](j_3, i_3) * taylorWindow(ant);
                } else {
                    win_xcube(ant) = 0.0;
                }
            }
            
            // Perform FFT
            Eigen::VectorXcd fft_result = fftshift(fft_wrapper(win_xcube));
            
            // Store result in the tensor
            for (int ang = 0; ang < fft_Ang; ++ang) {
                AngData(j_3, ang, i_3) = fft_result(ang);
            }
        }
    }
}

// Calculate the power spectrum for CFAR
Eigen::MatrixXd calculate_power_spectrum(const Eigen::Tensor<std::complex<double>, 3>& AngData) {
    int Nr = AngData.dimension(0);
    int fft_Ang = AngData.dimension(1);
    int Nd = AngData.dimension(2);
    
    // Initialize the power matrix
    Eigen::MatrixXd Xpow_db = Eigen::MatrixXd::Zero(Nr, fft_Ang);
    
    // Sum power across Doppler dimension - equivalent to squeeze(sum(abs(AngData).^2, 3))
    for (int i = 0; i < Nr; ++i) {
        for (int j = 0; j < fft_Ang; ++j) {
            double power_sum = 0.0;
            for (int k = 0; k < Nd; ++k) {
                std::complex<double> val = AngData(i, j, k);
                // Calculate |z|² directly
                power_sum += std::norm(val);  // std::norm gives |z|²
            }
            Xpow_db(i, j) = power_sum;
        }
    }
    
    // Convert to dB - equivalent to 10*log10(Xpow_db)
    for (int i = 0; i < Nr; ++i) {
        for (int j = 0; j < fft_Ang; ++j) {
            if (Xpow_db(i, j) > 0) {
                Xpow_db(i, j) = 10.0 * std::log10(Xpow_db(i, j));
            } else {
                // Match MATLAB behavior for zeros: use a very low value instead of -Inf
                Xpow_db(i, j) = -std::numeric_limits<double>::max() / 2.0;
            }
        }
    }
    
    // Take absolute value - equivalent to Xpow_db_1 = abs(Xpow_db)
    return Xpow_db.cwiseAbs();
}

// Global Gnuplot pipes to maintain separate connections
FILE* gnuplotPipeRangeAngle = nullptr;
FILE* gnuplotPipeCFAR = nullptr;

// Initialize Gnuplot pipes once at program start
void initGnuplot() {
    // Initialize the Range-Angle plot pipe
    if (gnuplotPipeRangeAngle == nullptr) {
        #ifdef _WIN32
            gnuplotPipeRangeAngle = _popen("gnuplot", "w");
        #else
            gnuplotPipeRangeAngle = popen("gnuplot", "w");
        #endif

        if (!gnuplotPipeRangeAngle) {
            std::cerr << "Error: Could not open Gnuplot pipe." << std::endl;
            return;
        }
        
    }
    
    // Initialize the CFAR plot pipe
    if (gnuplotPipeCFAR == nullptr) {
        #ifdef _WIN32
            gnuplotPipeCFAR = _popen("gnuplot", "w");
        #else
            gnuplotPipeCFAR = popen("gnuplot", "w");
        #endif
        
        if (!gnuplotPipeCFAR) {
            std::cerr << "Error: Could not open Gnuplot pipe." << std::endl;
            return;
        }
    }
}

// Close Gnuplot when done with all displays
void closeGnuplot() {
    if (gnuplotPipeRangeAngle) {
        #ifdef _WIN32
            _pclose(gnuplotPipeRangeAngle);
        #else
            pclose(gnuplotPipeRangeAngle);
        #endif
        gnuplotPipeRangeAngle = nullptr;
    }
    
    if (gnuplotPipeCFAR) {
        #ifdef _WIN32
            _pclose(gnuplotPipeCFAR);
        #else
            pclose(gnuplotPipeCFAR);
        #endif
        gnuplotPipeCFAR = nullptr;
    }
    
    // Clean up temporary files
    system("rm -f temp_range_angle_data.txt temp_cfar_data.txt");
}

// Display Range-Angle Map
void displayRangeAngleMap(const std::vector<double>& rng_grid, 
                          const std::vector<double>& angle_grid, 
                          const Eigen::MatrixXd& Xpow_db, 
                          int frameNumber) {

    // Ensure Gnuplot is initialized
    if (gnuplotPipeRangeAngle == nullptr) {
        initGnuplot();
        if (!gnuplotPipeRangeAngle) {
            std::cerr << "Error: Range-Angle Gnuplot pipe not available." << std::endl;
            return;
        }
    }

    // Check if the data is empty
    if (Xpow_db.rows() == 0 || Xpow_db.cols() == 0) {
        std::cerr << "Error: Xpow_db is empty." << std::endl;
        return;
    }

    // Always use the same file name to overwrite previous data
    std::string dataFileName = "temp_range_angle_data.txt";
    std::ofstream dataFile(dataFileName);
    if (!dataFile.is_open()) {
        std::cerr << "Error: Could not open temporary file for Range-Angle data." << std::endl;
        return;
    }

    // Write data to file with proper coordinates
    for (int i = 0; i < Xpow_db.rows(); ++i) {
        for (int j = 0; j < Xpow_db.cols(); ++j) {
            double range = (i < rng_grid.size()) ? rng_grid[i] : i;
            double angle = (j < angle_grid.size()) ? angle_grid[j] : j;
            dataFile << range << " " << angle << " " << Xpow_db(i, j) << std::endl;
        }
        dataFile << std::endl; // Empty line for gnuplot
    }
    dataFile.close();

     // Initialize the Range-Angle window
     fprintf(gnuplotPipeRangeAngle, "set terminal qt 0 size 800,600 title 'Range-Angle Map'\n");
     fprintf(gnuplotPipeRangeAngle, "set pm3d map\n");
     fprintf(gnuplotPipeRangeAngle, "set palette defined (0 'blue', 1 'cyan', 2 'yellow', 3 'red')\n");
     fprintf(gnuplotPipeRangeAngle, "set xlabel 'Range (meters)'\n");
     fprintf(gnuplotPipeRangeAngle, "set ylabel 'Angle of Arrival (degrees)'\n");
     fflush(gnuplotPipeRangeAngle);
    fprintf(gnuplotPipeRangeAngle, "set title 'MTI Range-Angle Map for Frame %d'\n", frameNumber);

    // Set x and y ranges if needed
    if (!rng_grid.empty()) {
        fprintf(gnuplotPipeRangeAngle, "set xrange [%f:%f]\n", rng_grid.front(), rng_grid.back());
    }
    if (!angle_grid.empty()) {
        fprintf(gnuplotPipeRangeAngle, "set yrange [%f:%f]\n", angle_grid.front(), angle_grid.back());
    }

    // Replot with the updated data
    fprintf(gnuplotPipeRangeAngle, "splot '%s' using 1:2:3 with image\n", dataFileName.c_str());
    fprintf(gnuplotPipeRangeAngle, "pause 0.1\n"); // Small pause to allow the display to update
    fflush(gnuplotPipeRangeAngle);

    std::cout << "Range-Angle Map Frame " << frameNumber << " displayed" << std::endl;
}

// Display CFAR Detection Map
void displayCFARMap(const std::vector<double>& rng_grid, 
                    const std::vector<double>& angle_grid, 
                    const Eigen::MatrixXd& cfar_signal, 
                    double offset,
                    int frameNumber) {

    // Ensure Gnuplot is initialized
    if (gnuplotPipeCFAR == nullptr) {
        initGnuplot();
        if (!gnuplotPipeCFAR) {
            std::cerr << "Error: CFAR Gnuplot pipe not available." << std::endl;
            return;
        }
    }

    // Check if the data is empty
    if (cfar_signal.rows() == 0 || cfar_signal.cols() == 0) {
        std::cerr << "Error: cfar_signal is empty." << std::endl;
        return;
    }

    // Always use the same file name to overwrite previous data
    std::string dataFileName = "temp_cfar_data.txt";
    std::ofstream dataFile(dataFileName);
    if (!dataFile.is_open()) {
        std::cerr << "Error: Could not open temporary file for CFAR data." << std::endl;
        return;
    }

    // Write data to file with proper coordinates
    
    for (int i = 0; i < cfar_signal.rows(); ++i) {
        for (int j = 0; j < cfar_signal.cols(); ++j) {
            double range = (i < rng_grid.size()) ? rng_grid[i] : i;
            double angle = (j < angle_grid.size()) ? angle_grid[j] : j;
            // Use j (column) as x-axis and i (row) as y-axis to match the transpose
            dataFile << range << " " << angle << " " << cfar_signal(i, j) << std::endl;
        }
        dataFile << std::endl; // Empty line for gnuplot
    }
    dataFile.close();

    // title to show current frame number and offset
        fprintf(gnuplotPipeCFAR, "set terminal qt 1 size 800,600 title 'CFAR Detection Map'\n");
            fprintf(gnuplotPipeCFAR, "set pm3d map\n");
            fprintf(gnuplotPipeCFAR, "set palette defined (0 'blue', 1 'cyan', 2 'yellow', 3 'red')\n");
            fprintf(gnuplotPipeCFAR, "set xlabel 'Range (meters)'\n");
            fprintf(gnuplotPipeCFAR, "set ylabel 'Angle of Arrival (degrees)'\n");
            fprintf(gnuplotPipeCFAR, "set cbrange [0:1]\n"); // Binary display
            fprintf(gnuplotPipeCFAR, "set title '1D CFAR Detection (Offset = %.1f dB) for Frame %d'\n", offset, frameNumber);
            fflush(gnuplotPipeCFAR);

    // Set x and y ranges if needed
    if (!rng_grid.empty()) {
        fprintf(gnuplotPipeCFAR, "set xrange [%f:%f]\n", rng_grid.front(), rng_grid.back());
    }
    if (!angle_grid.empty()) {
        fprintf(gnuplotPipeCFAR, "set yrange [%f:%f]\n", angle_grid.front(), angle_grid.back());
    }

    // Replot with the updated data
    fprintf(gnuplotPipeCFAR, "set view map\n");
    fprintf(gnuplotPipeCFAR, "splot '%s' using 1:2:3 with image\n", dataFileName.c_str());
    fprintf(gnuplotPipeCFAR, "pause 0.1\n"); // Small pause to allow the display to update
    fflush(gnuplotPipeCFAR);

    std::cout << "CFAR Detection Map Frame " << frameNumber << " displayed" << std::endl;
}

// CFAR Detection function
Eigen::MatrixXd performCFARDetection(const Eigen::MatrixXd& Xpow_db, int Tr, int Gr, double offset) {
    int numRows = Xpow_db.rows();
    int numCols = Xpow_db.cols();
    
    // Initialize CFAR signal matrix with zeros
    Eigen::MatrixXd cfar_signal = Eigen::MatrixXd::Zero(numRows, numCols);
    
    // Convert dB to power for CFAR processing
    auto db2pow = [](double db_value) -> double {
        return std::pow(10.0, db_value / 10.0);
    };
    
    // Convert power to dB
    auto pow2db = [](double power) -> double {
        return 10.0 * std::log10(power);
    };
    
    // Apply CFAR detection across range and angle dimensions
    for (int angle_idx = 0; angle_idx < numCols; angle_idx++) {
        for (int r = Tr + Gr + 1; r < numRows - (Tr + Gr); r++) {
            // Create vectors for training data
            std::vector<double> training_data;
            
            // Get leading training cells
            for (int i = r - Tr - Gr; i < r - Gr; i++) {
                training_data.push_back(Xpow_db(i, angle_idx));
            }
            
            // Get lagging training cells
            for (int i = r + Gr + 1; i <= r + Tr + Gr; i++) {
                training_data.push_back(Xpow_db(i, angle_idx));
            }
            
            // Calculate noise level from training cells (convert from dB)
            double noise_level = 0.0;
            for (double val : training_data) {
                noise_level += db2pow(val);
            }
            noise_level /= training_data.size();
            
            // Calculate threshold (convert back to dB and add offset)
            double threshold = pow2db(noise_level) + offset;
            
            // Get Cell Under Test (CUT)
            double CUT = Xpow_db(r, angle_idx);
            
            // Apply threshold test
            if (CUT > threshold) {
                cfar_signal(r, angle_idx) = 1.0;
            }
        }
    }
    
    return cfar_signal;
}

// displayRangeAngleMap 
void displayRangeAngleMapWithCFAR(
    const std::vector<double>& rng_grid, 
    const std::vector<double>& angle_grid, 
    const Eigen::MatrixXd& Xpow_db,
    const Eigen::MatrixXd& cfar_detections,
    int frameNumber) {

    // Ensure Gnuplot is initialized
    if (gnuplotPipeCFAR == nullptr) {
        initGnuplot();
        if (!gnuplotPipeCFAR) {
            std::cerr << "Error: Gnuplot pipe not available." << std::endl;
            return;
        }
    }

    // Check if the data is empty
    if (Xpow_db.rows() == 0 || Xpow_db.cols() == 0) {
        std::cerr << "Error: Xpow_db is empty." << std::endl;
        return;
    }

    // Create a combined visualization matrix
    Eigen::MatrixXd visualMatrix = Eigen::MatrixXd::Zero(Xpow_db.rows(), Xpow_db.cols());
    

    double minVal = Xpow_db.minCoeff();
    double maxVal = Xpow_db.maxCoeff();
    double range = maxVal - minVal;
    
    if (range > 0) {
        // Normalize to range [0, 0.05] to create a blue background
        for (int i = 0; i < Xpow_db.rows(); ++i) {
            for (int j = 0; j < Xpow_db.cols(); ++j) {
                visualMatrix(i, j) = 0.05 * (Xpow_db(i, j) - minVal) / range;
            }
        }
    }
    
    // Overlay CFAR detections with value 1.0 (yellow)
    for (int i = 0; i < cfar_detections.rows(); ++i) {
        for (int j = 0; j < cfar_detections.cols(); ++j) {
            if (cfar_detections(i, j) > 0.5) {
                visualMatrix(i, j) = 1.0;  // Set to maximum value (yellow)
            }
        }
    }

    // Create data file for the combined visualization
    std::string dataFileName = "temp_range_angle_cfar.txt";
    std::ofstream dataFile(dataFileName);
    if (!dataFile.is_open()) {
        std::cerr << "Error: Could not open temporary file for visualization data." << std::endl;
        return;
    }

    // Write visualization data to file
    for (int i = 0; i < visualMatrix.rows(); ++i) {
        for (int j = 0; j < visualMatrix.cols(); ++j) {
            double range = (i < rng_grid.size()) ? rng_grid[i] : i;
            double angle = (j < angle_grid.size()) ? angle_grid[j] : j;
            dataFile << range << " " << angle << " " << visualMatrix(i, j) << std::endl;
        }
        dataFile << std::endl; // Empty line for gnuplot
    }
    dataFile.close();

    // Update the plot title and display settings
    fprintf(gnuplotPipeCFAR, "set terminal qt 1 size 800,600 title 'CFAR Detection Map'\n");
    fprintf(gnuplotPipeCFAR, "set title ' CFAR Detection '\n");
    fprintf(gnuplotPipeCFAR, "set xlabel 'Range (meters)'\n");
    fprintf(gnuplotPipeCFAR, "set ylabel 'Angle (degrees)'\n");
    fprintf(gnuplotPipeCFAR, "set cblabel 'Normalized Power'\n");
    
    // Set the color palette to match the image (blue to yellow)
    fprintf(gnuplotPipeCFAR, "set palette defined (0 'dark-blue', 0.1 'blue', 0.5 'cyan', 0.7 'yellow', 1 'yellow')\n");
    fprintf(gnuplotPipeCFAR, "set cbrange [0:1]\n");
    fprintf(gnuplotPipeCFAR, "unset key\n");
    
    // Set the aspect ratio to match the visualization
    fprintf(gnuplotPipeCFAR, "set size ratio 0.75\n");
    
    // If range and angle grids are provided, set the plot ranges
    if (!rng_grid.empty() && !angle_grid.empty()) {
        fprintf(gnuplotPipeCFAR, "set xrange [%f:%f]\n", rng_grid.front(), rng_grid.back());
        fprintf(gnuplotPipeCFAR, "set yrange [%f:%f]\n", angle_grid.front(), angle_grid.back());
    }

    // Plot the combined visualization as a heatmap
    fprintf(gnuplotPipeCFAR, "plot '%s' using 1:2:3 with image notitle\n", dataFileName.c_str());

    fprintf(gnuplotPipeCFAR, "pause 0.5\n");
    fflush(gnuplotPipeCFAR);

    std::cout << "Range-Angle Map with CFAR Detections Frame " << frameNumber << " displayed" << std::endl;
}

// function prototypes 
double getRangeResolution();
std::vector<double> createTimeVector(int samples, double sampleRate);
Eigen::VectorXcd fft_wrapper(const Eigen::VectorXcd& input);
void process_doppler(const std::vector<Eigen::MatrixXcd>& rangeData_odd,
                    const std::vector<Eigen::MatrixXcd>& rangeData_even,
                    int Ne, int Nr, int Nd,
                    std::vector<Eigen::MatrixXcd>& dopData_odd,
                    std::vector<Eigen::MatrixXcd>& dopData_even,
                    std::vector<Eigen::MatrixXcd>& angle_input);
void process_angle_fft(const std::vector<Eigen::MatrixXcd>& angle_input,
                      int Nr, int Nd, int fft_Ang,
                      Eigen::Tensor<std::complex<double>, 3>& angData);
Eigen::MatrixXd calculate_power_spectrum(const Eigen::Tensor<std::complex<double>, 3>& angData);
void displayRangeAngleMap(const std::vector<double>& rng_grid,
                         const std::vector<double>& angle_grid,
                         const Eigen::MatrixXd& Xpow_db, int frameNum);
Eigen::MatrixXd performCFARDetection(const Eigen::MatrixXd& Xpow_db,
                                    int Tr, int Gr, double offset);
void displayRangeAngleMapWithCFAR(const std::vector<double>& rng_grid,
                                 const std::vector<double>& angle_grid,
                                 const Eigen::MatrixXd& Xpow_db,
                                 const Eigen::MatrixXd& cfar_signal,
                                 int frameNum);

// Define Point structure globally so it can be used in other functions
struct Point {
    double x, y;
};

void plotClusteringResult(const std::vector<Point>& points, const std::vector<int>& IDX, double epsilon, int minPts) {
    int k = 0;
    if (!IDX.empty()) {
        k = *std::max_element(IDX.begin(), IDX.end());
    }
    
    // Create temporary data files for each cluster and noise
    std::vector<std::string> filenames;
    std::vector<std::string> legends;
    
    // Create data file for noise points (IDX = 0)
    std::string noise_file = "cluster_noise.dat";
    std::ofstream noise_out(noise_file);
    bool has_noise = false;
    
    for (size_t i = 0; i < points.size(); ++i) {
        if (IDX[i] == 0) {
            noise_out << points[i].x << " " << points[i].y << std::endl;
            has_noise = true;
        }
    }
    noise_out.close();
    
    if (has_noise) {
        filenames.push_back(noise_file);
        legends.push_back("Noise");
    }
    
    // Create data files for each cluster
    for (int cluster = 1; cluster <= k; ++cluster) {
        std::string cluster_file = "cluster_" + std::to_string(cluster) + ".dat";
        std::ofstream cluster_out(cluster_file);
        bool has_points = false;
        
        for (size_t i = 0; i < points.size(); ++i) {
            if (IDX[i] == cluster) {
                cluster_out << points[i].x << " " << points[i].y << std::endl;
                has_points = true;
            }
        }
        cluster_out.close();
        
        if (has_points) {
            filenames.push_back(cluster_file);
            legends.push_back("Cluster #" + std::to_string(cluster));
        }
    }
    
    // Create gnuplot pipe for direct display
    FILE* gnuplotPipe = popen("gnuplot -persist", "w");
    if (!gnuplotPipe) {
        std::cerr << "Error opening gnuplot pipe!" << std::endl;
        return;
    }
    
    // Set up gnuplot parameters
    fprintf(gnuplotPipe, "set title 'DBSCAN Clustering (epsilon = %f, MinPts = %d)'\n", epsilon, minPts);
    fprintf(gnuplotPipe, "set xlabel 'X (m)'\n");
    fprintf(gnuplotPipe, "set ylabel 'Y (m)'\n");
    fprintf(gnuplotPipe, "set size ratio 1\n");
    fprintf(gnuplotPipe, "set grid\n");
    fprintf(gnuplotPipe, "set key outside right\n");
    
    // Build the plot command
    std::string plotCmd = "plot ";
    for (size_t i = 0; i < filenames.size(); ++i) {
        if (i > 0) plotCmd += ", ";
        
        if (filenames[i] == "cluster_noise.dat") {
            plotCmd += "'" + filenames[i] + "' with points pt 6 ps 1 lc rgb 'black' title '" + legends[i] + "'";
        } else {
            int color_index = std::stoi(filenames[i].substr(8, filenames[i].find('.') - 8));
            plotCmd += "'" + filenames[i] + "' with points pt 2 ps 1.5 lc " + std::to_string(color_index) + " title '" + legends[i] + "'";
        }
    }
    plotCmd += "\n";
    
    // Execute plot command
    fprintf(gnuplotPipe, "%s", plotCmd.c_str());
    fflush(gnuplotPipe);
    
    std::cout << "Cluster plot displayed in gnuplot window" << std::endl;
    }

bool saveRadarDataToSingleFile(const FrameData& data, const std::string& outputFile = "radar_data_selected.csv") {
    std::ofstream file(outputFile);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << outputFile << " for writing." << std::endl;
        return false;
    }
    
    // Write file header
    file << "# Radar Data Export - c++\n";
    auto now = std::chrono::system_clock::now();
    std::time_t time_now = std::chrono::system_clock::to_time_t(now);
    file << "# Date: " << std::put_time(std::localtime(&time_now), "%Y-%m-%d %H:%M:%S") << "\n\n";
    
    // Write data in a single section with the requested fields
    file << "SECTION,STRUCTURE_FIELDS\n";
    file << "Frame,Mean1_Theta(rad),Mean1_Rho(m),Detections,Range(m)\n";
    
    // Save all 200 frames to match the MATLAB output
    const size_t expected_frames = 200;
    for (size_t i = 0; i < std::min(data.frames.size(), expected_frames); ++i) {
        // Frames (complex number, split into real and imaginary parts)
        double frame_real = 0.0, frame_imag = 0.0;
        if (i < data.frames.size() && data.frames[i].size() > 0) {
            frame_real = data.frames[i](0, 0).real();
            frame_imag = data.frames[i](0, 0).imag();
        }
        
        // Mean1 (theta, rho)
        double theta = (i < data.mean1.size()) ? data.mean1[i].first : std::numeric_limits<double>::quiet_NaN();
        double rho = (i < data.mean1.size()) ? data.mean1[i].second : std::numeric_limits<double>::quiet_NaN();
        
        // Detections
        int detections = (i < data.detections.size()) ? data.detections[i] : 0;
        
        // Range (from range_angle_pairs, take the first range if available)
        double range = 0.0;
        if (i < data.range_angle_pairs.size() && !data.range_angle_pairs[i].empty()) {
            range = data.range_angle_pairs[i][0].range;
        }
        
        // Write the row
        file << i << "," 
             << (std::isnan(theta) ? "NaN" : std::to_string(theta)) << "," 
             << (std::isnan(rho) ? "NaN" : std::to_string(rho)) << "," 
             << detections << "," 
             << range << "\n";
    }
    
    file.close();
    std::cout << "Selected radar data successfully saved to " << outputFile << std::endl;
    return true;
}

// Main function
int main() {
    // Read radar data
    Eigen::MatrixXcd retVal = readDCA1000("sr_car.bin");
    if (retVal.size() == 0) {
        std::cerr << "Error: Data not read correctly." << std::endl;
        return -1;
    }

    // Frame parameters
    int frames = 200;
    int totalCols = retVal.cols();
    int s = totalCols / frames;           // samples per frame
    int m_chirps = s / ADC_Samples;       // chirps per frame
    int ant_ele = retVal.rows();          // number of receivers

    // Debugging output
    std::cout << "retVal dimensions: " << retVal.rows() << " x " << retVal.cols() << std::endl;
    std::cout << "Frames: " << frames << ", Samples per frame: " << s << ", Chirps per frame: " << m_chirps << std::endl;

    // Extract receiver data (assuming at least 4 receivers)
    Eigen::RowVectorXcd Receiver_1 = retVal.row(0);
    Eigen::RowVectorXcd Receiver_2 = retVal.row(1);
    Eigen::RowVectorXcd Receiver_3 = retVal.row(2);
    Eigen::RowVectorXcd Receiver_4 = retVal.row(3);

    double rangeres = getRangeResolution();

    // Generate Reference Chirp
    std::vector<double> t = createTimeVector(ADC_Samples, SampleRate);
    Eigen::RowVectorXcd reference_chirp(ADC_Samples);
    for (int i = 0; i < ADC_Samples; ++i) {
        double phase = M_PI * Freq_Slope * t[i] * t[i];
        reference_chirp(i) = std::exp(std::complex<double>(0, phase));
    }

    // Initialize before the loop
    double int_time = 107.3e-3;
    double time = 0;

    // Create a FrameData instance
    FrameData data;

    std::vector<Point> finalPoints;
    std::vector<int> finalClusterIDs;

    // Apply DBSCAN clustering
    double epsilon = 1.0;  
    int minPts = 2;        

    // Loop through frames
    for (int f = 0; f < frames; ++f) {
        int startCol = f * s;
        // Current frame data: [antennas x s]
        Eigen::MatrixXcd data_frame = retVal.block(0, startCol, ant_ele, s);

        // Split frame into chirps: each chirp is ADC_Samples columns
        std::vector<Eigen::MatrixXcd> data_chirp;
        for (int nC = 0; nC < m_chirps; ++nC) {
            int col0 = nC * ADC_Samples;
            if (col0 + ADC_Samples <= data_frame.cols()) {
                data_chirp.push_back(data_frame.block(0, col0, ant_ele, ADC_Samples));
            }
        }

        // Separate odd and even chirps
        std::vector<Eigen::MatrixXcd> chirp_odd, chirp_even;
        for (size_t nC = 0; nC < data_chirp.size(); ++nC) {
            if (nC % 2 == 0)
                chirp_odd.push_back(data_chirp[nC]);
            else
                chirp_even.push_back(data_chirp[nC]);
        }

        // Permute chirp matrices: make rows = samples, columns = receivers
        for (auto &mat : chirp_odd)
            mat.transposeInPlace();
        for (auto &mat : chirp_even)
            mat.transposeInPlace();

        // Apply MTI (Moving Target Indicator)
        std::vector<Eigen::MatrixXcd> chirp_odd_MTI, chirp_even_MTI;
        for (size_t i = 1; i < chirp_odd.size(); ++i) {
            chirp_odd_MTI.push_back(chirp_odd[i] - chirp_odd[i-1]);
        }
        for (size_t i = 1; i < chirp_even.size(); ++i) {
            chirp_even_MTI.push_back(chirp_even[i] - chirp_even[i-1]);
        }

        // Verify chirp counts
        if (chirp_odd_MTI.size() != chirp_even_MTI.size()) {
            std::cerr << "Error: Mismatch in odd and even chirp counts after MTI" << std::endl;
            return -1;
        }

        // Apply Range FFT
        int Nr = chirp_odd_MTI[0].rows();    // Number of ADC samples (256)
        int Ne = chirp_odd_MTI[0].cols();    // Number of antennas/receivers
        int Nd = chirp_odd_MTI.size();       // Number of chirps after MTI (corrected)

        // Create 3D arrays for range FFT results
        std::vector<Eigen::MatrixXcd> Rangedata_odd;
        std::vector<Eigen::MatrixXcd> Rangedata_even;

        // Initialize with correct dimensions
        for (int i = 0; i < Ne; ++i) {
            Rangedata_odd.push_back(Eigen::MatrixXcd::Zero(Nr, Nd));
            Rangedata_even.push_back(Eigen::MatrixXcd::Zero(Nr, Nd));
        }

        // For odd chirps
        for (int i_1 = 0; i_1 < Ne; ++i_1) {
            for (int j_1 = 0; j_1 < Nd; ++j_1) {
                Eigen::VectorXcd column_data(Nr);
                for (int r = 0; r < Nr; ++r) {
                    column_data(r) = chirp_odd_MTI[j_1](r, i_1);
                }
                Eigen::VectorXcd fft_result = fft_wrapper(column_data);
                // Copy results element by element
                for (int r = 0; r < Nr; ++r) {
                    Rangedata_odd[i_1](r, j_1) = fft_result(r);
                }
            }
        }

        // For even chirps - avoid assigning entire columns
        for (int i_1 = 0; i_1 < Ne; ++i_1) {
            for (int j_1 = 0; j_1 < Nd; ++j_1) {
                Eigen::VectorXcd column_data(Nr);
                for (int r = 0; r < Nr; ++r) {
                    column_data(r) = chirp_even_MTI[j_1](r, i_1);
                }
                Eigen::VectorXcd fft_result = fft_wrapper(column_data);
                // Copy results element by element
                for (int r = 0; r < Nr; ++r) {
                    Rangedata_even[i_1](r, j_1) = fft_result(r);
                }
            }
        }

        std::cout << "Starting Doppler processing with Ne=" << Ne << ", Nr=" << Nr << ", Nd=" << Nd << std::endl;

        // Apply Doppler FFT and virtual antenna combination
        std::vector<Eigen::MatrixXcd> DopData_odd, DopData_even, Angle_input;
        process_doppler(Rangedata_odd, Rangedata_even, Ne, Nr, Nd, DopData_odd, DopData_even, Angle_input);

        // Store results in FrameData 
        data.frames.push_back(Angle_input[0]); // Example: store first virtual antenna
        data.time.push_back(time);
        time += int_time;
        
        // Angle FFT parameters
        int fft_Ang = 8; // 8 virtual antennas for angle FFT
        
        // Create tensor for angle FFT output
        Eigen::Tensor<std::complex<double>, 3> AngData;
        
        // Perform angle FFT
        process_angle_fft(Angle_input, Nr, Nd, fft_Ang, AngData);
        
        // Create angle and range grids
        double FoV = 60.0; // Field of view in degrees
        std::vector<double> angle_grid(fft_Ang);
        std::vector<double> w(fft_Ang);
        
        // Calculate angle grid
        for (int i = 0; i < fft_Ang; ++i) {
            double normalized_index = static_cast<double>(i) / (fft_Ang - 1) * 2.0 - 1.0;
            w[i] = -std::sin(FoV * M_PI / 180.0) * normalized_index;
            angle_grid[i] = std::asin(w[i]) * 180.0 / M_PI;
        }
        
        // Calculate range grid
        std::vector<double> rng_grid(Nr);
        for (int i = 0; i < Nr; ++i) {
            rng_grid[i] = i * rangeres;
        }
        
        // Calculate power spectrum for CFAR
        Eigen::MatrixXd Xpow_db = calculate_power_spectrum(AngData);
        
        if (f == 0) { // Save only the first frame for testing
            std::ofstream outFile("power_spectrum.txt");
            if (outFile.is_open()) {
                outFile << "# Range (m), Angle (deg), Power (dB)" << std::endl;
                for (int i = 0; i < Nr; ++i) {
                    for (int j = 0; j < fft_Ang; ++j) {
                        outFile << rng_grid[i] << " " << angle_grid[j] << " " << Xpow_db(i, j) << std::endl;
                    }
                    outFile << std::endl; 
                }
                outFile.close();
                std::cout << "Power spectrum saved to file." << std::endl;
            } else {
                std::cerr << "Failed to open output file." << std::endl;
            }
        }

        // Display Range-Angle Map
        displayRangeAngleMap(rng_grid, angle_grid, Xpow_db, f);

        // CFAR parameters
        int Tr = 10;      // Training cells in range dimension
        int Gr = 4;       // Guard cells in range dimension
        double offset = 5.5; // Offset for threshold

        // Perform CFAR detection
        Eigen::MatrixXd cfar_signal = performCFARDetection(Xpow_db, Tr, Gr, offset);
    
        // Display Range-Angle Map with CFAR detections
        displayRangeAngleMapWithCFAR(rng_grid, angle_grid, Xpow_db, cfar_signal, f);

        // Get CFAR detection positions
        std::vector<int> row_indices, col_indices;
        std::vector<double> detected_ranges, detected_angles_rad;

        // Find all positions where CFAR detected a target
        for (int i = 0; i < cfar_signal.rows(); ++i) {
            for (int j = 0; j < cfar_signal.cols(); ++j) {
                if (cfar_signal(i, j) == 1) {
                    row_indices.push_back(i);
                    col_indices.push_back(j);
                    detected_ranges.push_back(rng_grid[i]);
                    // Convert angle to radians for processing
                    detected_angles_rad.push_back(angle_grid[j] * M_PI / 180.0);
                }
            }
        }

        int num_detections = row_indices.size();
        std::cout << "Number of detections: " << num_detections << std::endl;

        // Store detections in frame data structure
        data.detections.push_back(num_detections);

        std::vector<RangeAnglePair> range_angle_pairs;

        for (int i = 0; i < num_detections; ++i) {
            RangeAnglePair pair = {detected_ranges[i], detected_angles_rad[i]};
            range_angle_pairs.push_back(pair);
        }
        data.range_angle_pairs.push_back(range_angle_pairs);

        std::vector<Point> points;

        for (int i = 0; i < num_detections; ++i) {
            double x = detected_ranges[i] * cos(detected_angles_rad[i]);
            double y = detected_ranges[i] * sin(detected_angles_rad[i]);
            points.push_back({x, y});
        }

        // DBSCAN implementation
        class DBSCAN {
        private:
            std::vector<Point> points;
            double epsilon;
            int minPts;
            std::vector<int> cluster_ids;
            std::vector<bool> visited;
            std::vector<bool> is_noise;
    
            std::vector<int> regionQuery(int pointIdx) {
                std::vector<int> neighbors;
                for (size_t i = 0; i < points.size(); ++i) {
                    double distance = sqrt(pow(points[i].x - points[pointIdx].x, 2) + 
                                          pow(points[i].y - points[pointIdx].y, 2));
                    if (distance <= epsilon) {
                        neighbors.push_back(i);
                    }
                }
                return neighbors;
            }
    
            void expandCluster(int pointIdx, std::vector<int>& neighbors, int clusterId) {
                cluster_ids[pointIdx] = clusterId;
        
                for (size_t i = 0; i < neighbors.size(); ++i) {
                    int neighborIdx = neighbors[i];
            
                    if (!visited[neighborIdx]) {
                        visited[neighborIdx] = true;
                
                        std::vector<int> neighborNeighbors = regionQuery(neighborIdx);
                        if (neighborNeighbors.size() >= static_cast<size_t>(minPts)) {
                            // Add neighborNeighbors to neighbors
                            for (int nn : neighborNeighbors) {
                                if (std::find(neighbors.begin(), neighbors.end(), nn) == neighbors.end()) {
                                    neighbors.push_back(nn);
                                }
                            }
                        }
                    }
            
                    if (cluster_ids[neighborIdx] == 0) {
                        cluster_ids[neighborIdx] = clusterId;
                    }
                }
            }
    
        public:
            DBSCAN(const std::vector<Point>& pts, double eps, int min_pts) 
                : points(pts), epsilon(eps), minPts(min_pts) {
                cluster_ids = std::vector<int>(points.size(), 0);
                visited = std::vector<bool>(points.size(), false);
                is_noise = std::vector<bool>(points.size(), false);
            }
    
            std::vector<int> run() {
                int clusterId = 0;
        
                for (size_t i = 0; i < points.size(); ++i) {
                    if (visited[i]) continue;
            
                    visited[i] = true;
                    std::vector<int> neighbors = regionQuery(i);
            
                    if (neighbors.size() < static_cast<size_t>(minPts)) {
                        // Mark as noise
                        is_noise[i] = true;
                    } else {
                        // Start a new cluster
                        clusterId++;
                        expandCluster(i, neighbors, clusterId);
                    }
                }
        
                return cluster_ids;
            }
    
            std::vector<bool> getNoiseStatus() const {
                return is_noise;
            }
        };

        DBSCAN dbscan(points, epsilon, minPts);
        std::vector<int> cluster_ids = dbscan.run();
        std::vector<bool> is_noise = dbscan.getNoiseStatus();

        // Process clustering results
        if (num_detections == 1) {
            // Only one detection - directly assign it as the mean
            data.mean1.push_back({detected_angles_rad[0], detected_ranges[0]});
            data.flag1.push_back(false);
        } else if (num_detections > 1) {
            // Handle multiple detections with clustering
    
            // Track clusters - for now we'll just focus on cluster 1
            double sum_x1 = 0.0, sum_y1 = 0.0;
            int count1 = 0;
    
            std::vector<NoisePoint> noise_points;
    
            for (int i = 0; i < num_detections; ++i) {
                if (cluster_ids[i] == 1) {
                    sum_x1 += points[i].x;
                    sum_y1 += points[i].y;
                    count1++;
                } else if (cluster_ids[i] == 0) {
                    // This is a noise point
                    noise_points.push_back({detected_angles_rad[i], detected_ranges[i]});
                }
            }
    
            // Store noise points
            data.noise.push_back(noise_points);
    
            // Compute mean of cluster 1 if it exists
            if (count1 > 0) {
                double mean_x1 = sum_x1 / count1;
                double mean_y1 = sum_y1 / count1;
        
                // Convert Cartesian back to polar
                double theta1 = atan2(mean_y1, mean_x1);
                double rho1 = sqrt(mean_x1*mean_x1 + mean_y1*mean_y1);

                theta=theta1;
                rho=rho1;
        
                data.mean1.push_back({theta1, rho1});
                data.flag1.push_back(false);
            } else {
                // No valid cluster 1
                data.mean1.push_back({std::numeric_limits<double>::quiet_NaN(), 
                                     std::numeric_limits<double>::quiet_NaN()});
                data.flag1.push_back(true);
            }
        } else {
            // No detections
            data.mean1.push_back({std::numeric_limits<double>::quiet_NaN(), 
                                 std::numeric_limits<double>::quiet_NaN()});
            data.flag1.push_back(true);
        }

        // Update frame time
        time += int_time;
        data.time.push_back(time);

        // For the single detection case:
        if (num_detections == 1) {
            // Save single-point detection
            finalPoints.push_back(points[0]);
            finalClusterIDs.push_back(1);

            // Only one detection - directly assign it as the mean
            data.mean1.push_back({detected_angles_rad[0], detected_ranges[0]});
            data.flag1.push_back(false);

        } else if (num_detections > 1) {
            // Handle multiple detections with clustering

            // Track clusters
            double sum_x1 = 0.0, sum_y1 = 0.0;
            int count1 = 0;

            std::vector<NoisePoint> noise_points;

            for (int i = 0; i < num_detections; ++i) {
                if (cluster_ids[i] == 1) {
                    sum_x1 += points[i].x;
                    sum_y1 += points[i].y;
                    count1++;
                } else if (cluster_ids[i] == 0) {
                    // This is a noise point - store original polar coordinates
                    noise_points.push_back({detected_angles_rad[i], detected_ranges[i]});
                }

                // Collect all points and cluster IDs for final plot
                finalPoints.push_back(points[i]);
                finalClusterIDs.push_back(cluster_ids[i]);
            }

            // Store noise points
            data.noise.push_back(noise_points);

            // Compute mean of cluster 1 if it exists
            if (count1 > 0) {
                double mean_x1 = sum_x1 / count1;
                double mean_y1 = sum_y1 / count1;

                // Convert Cartesian back to polar
                double theta1 = atan2(mean_y1, mean_x1);
                double rho1 = sqrt(mean_x1*mean_x1 + mean_y1*mean_y1);

                data.mean1.push_back({theta1, rho1});
                data.flag1.push_back(false);
            } else {
                // No valid cluster 1
                data.mean1.push_back({std::numeric_limits<double>::quiet_NaN(), 
                                      std::numeric_limits<double>::quiet_NaN()});
                data.flag1.push_back(true);
            }

        } else {
            // No detections
            data.mean1.push_back({std::numeric_limits<double>::quiet_NaN(), 
                                  std::numeric_limits<double>::quiet_NaN()});
            data.flag1.push_back(true);
        }

        // Update and store time
        time += int_time;
        data.time.push_back(time);

        if (num_detections > 0) {
            finalPoints.insert(finalPoints.end(), points.begin(), points.end());
            finalClusterIDs.insert(finalClusterIDs.end(), cluster_ids.begin(), cluster_ids.end());
        }
        
        std::cout << "Frame " << f + 1 << " processed." << std::endl;
    }

    // Save the results
    if (saveRadarDataToSingleFile(data, "DBSCAN.csv")) {
        std::cout << "Selected radar data fields successfully saved." << std::endl;
    } else {
        std::cerr << "Error saving radar data." << std::endl;
    }

    if (!finalPoints.empty()) {
        plotClusteringResult(finalPoints, finalClusterIDs, epsilon, minPts);
    }

    std::cout << "Processing completed." << std::endl;
    return 0;
}