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
#include <iomanip> // Function to plot MTI data 
#include <chrono>
#include <thread>
#include <cstdio>



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

// Function declarations
void initGnuplot();
void displayRDMap(int nS, const Eigen::MatrixXd& abs_final_fft, int frameNumber);
void closeGnuplot();

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
    fprintf(gnuplotPipe, "set terminal qt size 800,600\n");
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
    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    
    std::cout << "RD Map Frame " << frameNumber << " displayed " << std::endl;
}

/*void closeGnuplot() {
    if (gnuplotPipe) {
        fprintf(gnuplotPipe, "exit\n");
        fflush(gnuplotPipe);
        
        #ifdef _WIN32
            _pclose(gnuplotPipe);
        #else
            pclose(gnuplotPipe);
        #endif
        
        gnuplotPipe = nullptr;
    }
}*/

void saveOutputSignalToFile(std::ofstream& file, const Eigen::MatrixXcd& final_fft) {
    file << std::fixed << std::setprecision(6);  
    file << "Output Signal Data :\n";
    
    for (int i = 0; i < final_fft.rows(); ++i) {
        for (int j = 0; j < final_fft.cols(); ++j) {
            file << final_fft(i, j).real() << " + " << final_fft(i, j).imag() << "i\t";
        }
        file << "\n";
    }
    
    file << std::endl; // Add an empty line between frames
    
    file.close();
    std::cout << "Output signal saved to file" << std::endl;
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
 std::cout << "Range-Doppler FFT Dimensions: " << abs_final_fft.rows() << " x " << abs_final_fft.cols() << std::endl;

 displayRDMap(nS, abs_final_fft,f);

 // Open file for output signal
std::ofstream outputSignalFile("fft_output.txt");
if (!outputSignalFile.is_open()) {
    std::cerr << "Error opening fft_output.txt file!" << std::endl;
    return -1;
}

// Save output signal to file
saveOutputSignalToFile(outputSignalFile, final_fft);
}
   
 //closeGnuplot();


    return 0;
}
