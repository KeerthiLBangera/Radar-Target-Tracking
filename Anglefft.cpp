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

// Include your DCA1000 reader 
#include "readDCA1000.h"

constexpr double c = 299792458;               // Speed of light (m/s)
constexpr double SampleRate = 5000e3;          // Sample rate (5e6 samples/s)
constexpr int ADC_Samples = 256;               // ADC samples per chirp

constexpr double Freq_Slope = 43.995e12;       // Chirp slope (Hz/s)
constexpr double idletime = 6e-6;
constexpr double rampendtime = 87.3e-6;
constexpr double carrier_frequency = 77e9;     // 77 GHz
constexpr double start_freq = carrier_frequency; // Starting frequency


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

// Replace your fft_wrapper function with this version
Eigen::VectorXcd fft_wrapper(const Eigen::VectorXcd& input) {
    // Make a copy instead of using assignment
    Eigen::VectorXcd result(input.size());
    for (int i = 0; i < input.size(); ++i) {
        result(i) = input(i);
    }
    custom_fft(result);
    return result;
}


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
    
    // Create Taylor window for angle FFT
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
    
    // Sum power across Doppler dimension
    for (int i = 0; i < Nr; ++i) {
        for (int j = 0; j < fft_Ang; ++j) {
            double power_sum = 0.0;
            for (int k = 0; k < Nd; ++k) {
                power_sum += std::abs(AngData(i, j, k)) * std::abs(AngData(i, j, k));
            }
            Xpow_db(i, j) = power_sum;
        }
    }
    
    // Convert to dB
    for (int i = 0; i < Nr; ++i) {
        for (int j = 0; j < fft_Ang; ++j) {
            if (Xpow_db(i, j) > 0) {
                Xpow_db(i, j) = 10.0 * std::log10(Xpow_db(i, j));
            } else {
                Xpow_db(i, j) = -200.0; // Set a floor for log of zero
            }
        }
    }
    
    return Xpow_db.cwiseAbs();
}

// Global Gnuplot pipe to maintain a single connection
FILE* gnuplotPipe = nullptr;

// Initialize Gnuplot pipe once at program start
void initGnuplot() {
    if (gnuplotPipe == nullptr) {
        #ifdef _WIN32
            gnuplotPipe = _popen("gnuplot", "w");
        #else
            gnuplotPipe = popen("gnuplot", "w");
        #endif
        
        if (gnuplotPipe) {
            // Initialize the window just once
            fprintf(gnuplotPipe, "set terminal qt size 800,600 title 'Range-Angle Map'\n");
            fprintf(gnuplotPipe, "set pm3d map\n");
            fprintf(gnuplotPipe, "set palette defined (0 'blue', 1 'cyan', 2 'yellow', 3 'red')\n");
            fprintf(gnuplotPipe, "set xlabel 'Range (meters)'\n");
            fprintf(gnuplotPipe, "set ylabel 'Angle of Arrival (degrees)'\n");
            fflush(gnuplotPipe);
        }
    }
}

// Close Gnuplot when done with all displays
void closeGnuplot() {
    if (gnuplotPipe) {
        #ifdef _WIN32
            _pclose(gnuplotPipe);
        #else
            pclose(gnuplotPipe);
        #endif
        gnuplotPipe = nullptr;
    }
    
    // Clean up temporary files
    system("rm -f temp_range_angle_data.txt");
}

void displayRangeAngleMap(const std::vector<double>& rng_grid, 
    const std::vector<double>& angle_grid, 
    const Eigen::MatrixXd& Xpow_db, 
    int frameNumber) {

    // Ensure Gnuplot is initialized
    if (gnuplotPipe == nullptr) {
        initGnuplot();
        if (!gnuplotPipe) {
            std::cerr << "Error: Gnuplot pipe not available." << std::endl;
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

    // Update the title to show current frame number
    fprintf(gnuplotPipe, "set title 'MTI Range-Angle Map for Frame %d'\n", frameNumber);

    // Set x and y ranges if needed
    if (!rng_grid.empty()) {
        fprintf(gnuplotPipe, "set xrange [%f:%f]\n", rng_grid.front(), rng_grid.back());
    }
    if (!angle_grid.empty()) {
        fprintf(gnuplotPipe, "set yrange [%f:%f]\n", angle_grid.front(), angle_grid.back());
    }

    // Replot with the updated data
    fprintf(gnuplotPipe, "splot '%s' using 1:2:3 with image\n", dataFileName.c_str());
    fprintf(gnuplotPipe, "pause 0.1\n"); // Small pause to allow the display to update
    fflush(gnuplotPipe);

    std::cout << "Range-Angle Map Frame " << frameNumber << " displayed" << std::endl;
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
    int frames = 100;
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

    // Define a struct to store the data
    struct FrameData {
        std::vector<Eigen::MatrixXcd> frames;
        std::vector<std::complex<double>> mean1;
        std::vector<std::complex<double>> mean2;
        std::vector<bool> flag1;
        std::vector<bool> flag2;
        std::vector<double> noise;
        std::vector<double> time;
    };

    // Create an instance of the struct
    FrameData data;

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
        int Nd = chirp_odd_MTI.size();       // Number of chirps after MTI

     

// Create 3D arrays for range FFT results
std::vector<Eigen::MatrixXcd> Rangedata_odd;
std::vector<Eigen::MatrixXcd> Rangedata_even;

// Initialize with correct dimensions
for (int i = 0; i < Ne; ++i) {
    Rangedata_odd.push_back(Eigen::MatrixXcd::Zero(Nr, Nd));
    Rangedata_even.push_back(Eigen::MatrixXcd::Zero(Nr, Nd));
}

// For odd chirps
// For odd chirps - avoid assigning entire columns
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

// For even chirps 
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

        // Store results in FrameData (optional, depending on your needs)
        data.frames.push_back(Angle_input[0]); // Example: store first virtual antenna
        data.time.push_back(time);
        time += int_time;
        


       // Apply Range FFT
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
        
        // Store results in FrameData (optional, depending on your needs)
        data.frames.push_back(Angle_input[0]); // Example: store first virtual antenna
        data.time.push_back(time);
        time += int_time;
        
       
        if (f == 0) { // Save only the first frame for testing
            std::ofstream outFile("power_spectrum.txt");
            if (outFile.is_open()) {
                outFile << "# Range (m), Angle (deg), Power (dB)" << std::endl;
                for (int i = 0; i < Nr; ++i) {
                    for (int j = 0; j < fft_Ang; ++j) {
                        outFile << rng_grid[i] << " " << angle_grid[j] << " " << Xpow_db(i, j) << std::endl;
                    }
                    outFile << std::endl; // Add blank line between range bins for gnuplot
                }
                outFile.close();
                std::cout << "Power spectrum saved to file." << std::endl;
            } else {
                std::cerr << "Failed to open output file." << std::endl;
            }
        }

        // Display Range-Angle Map
        displayRangeAngleMap(rng_grid, angle_grid, Xpow_db, f);
        
        std::cout << "Frame " << f + 1 << " processed." << std::endl;
    }

    std::cout << "Processing completed." << std::endl;
    return 0;
}