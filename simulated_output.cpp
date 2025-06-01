#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <Eigen/Dense>
#include "Track.h"
#include "Radar.h"
#include "parameters.h"
#include "radar_info.h"
#include "target_info.h"
#include "GNN_filter.h"
#include "main_GNN.h"
#include "simulate_target_truth.h"
#include "get_measurements_radar.h"

using namespace std;
using namespace Eigen;

struct TrackOut {
    vector<Vector2d> pos;
};
void generateGnuplotData(
    const vector<TrackOut>& track_out,
    const vector<Target>& X,
    const vector<vector<MeasurementRadar>>& Z
) {
    ofstream plotFile("plot_data_o.txt");

    // Track data
    for (const auto& track : track_out) {
        for (const auto& pt : track.pos) {
            if (std::isnan(pt(0)) || std::isnan(pt(1))) continue;
            if (pt.norm() < 50 || pt.norm() > 25000) continue;
            plotFile << pt(0) << " " << pt(1) << " T\n";
        }
        plotFile << "\n\n";
    }

    // True state data
    for (const auto& target : X) {
        for (int i = 0; i < target.state.cols(); ++i) {
            double x = target.state(0, i);
            double y = target.state(2, i);
            if (std::isnan(x) || std::isnan(y)) continue;
            if (x < -10000 || x > 20000 || y < -10000 || y > 20000) continue;
            plotFile << x << " " << y << " X\n";
        }
        plotFile << "\n\n";
    }

    // Measurement data
    for (const auto& scan : Z) {
        for (const auto& meas : scan) {
            for (const auto& detection : meas.detections) {
                if (detection.size() < 2) continue;
                double range = detection[0], azimuth = detection[1];
                if (range < 50 || range > 25000 || std::isnan(range)) continue;
                if (azimuth < -M_PI / 55 || azimuth > M_PI / 55 || std::isnan(azimuth)) continue;
                double x = range * cos(azimuth);
                double y = range * sin(azimuth);
                plotFile << x << " " << y << " M\n";
            }
        }
        plotFile << "\n";
    }

    plotFile.close();

    // Gnuplot script with interactive features
    ofstream script("plot_script_o.gnu");
    script << "set title 'Radar Tracking Visualization'\n";
    script << "set xlabel 'X (meters)'\n";
    script << "set ylabel 'Y (meters)'\n";
    script << "set grid\n";
    script << "set xrange [-1000:15000]\n";
    script << "set yrange [-1000:15000]\n";
    script << "set key outside\n";
    
    // Enable mouse interaction
    script << "set mouse mouseformat 3\n";
    script << "set mouse zoomcoordinates\n";
    script << "set mouse zoomfactors 1.1, 1.1\n";
    
    // Add functionality hints to plot title
    script << "set title 'Radar Tracking Visualization\\nLeft-click+drag: zoom, Middle-click: restore view, Right-click: context menu'\n";
    
    // Enable persistent mode and responsive terminal
    script << "set terminal wxt enhanced size 1200,800 font 'Arial,10' persist raise\n";
    
    // Add a reset button
    script << "bind \"r\" \"set xrange [-1000:15000]; set yrange [-1000:15000]; replot\"\n";
    
    script << "plot \\\n";
    script << "'plot_data_o.txt' using 1:2:(strcol(3) eq \"T\" ? 1 : 0) with lines lc rgb 'blue' title 'Tracks', \\\n";
    script << "'plot_data_o.txt' using 1:2:(strcol(3) eq \"X\" ? 1 : 0) with points pt 4 ps 1.5 lc rgb 'green' title 'True States', \\\n";
    script << "'plot_data_o.txt' using 1:2:(strcol(3) eq \"M\" ? 1 : 0) with points pt 7 ps 1.0 lc rgb 'red' title 'Measurements'\n";
    
    // Add interactive hint message
    script << "print \"\\n=== INTERACTIVE CONTROLS ===\\n\"\n";
    script << "print \"- Left-click and drag: zoom into selected region\"\n";
    script << "print \"- Middle-click: reset to original view\"\n";
    script << "print \"- Right-click: open context menu with more options\"\n";
    script << "print \"- Press 'r' key: reset zoom to original range\"\n";
    script << "print \"- Mouse wheel: zoom in/out\"\n";
    script << "print \"==============================\\n\"\n";
    
    script.close();

    // Use a system call that keeps the window open
    system("gnuplot plot_script_o.gnu");
}

int main() {
    vector<Sensor> Sensors = radar_info();
    vector<MonteCarloData> monteCarloResults = main_GNN(Sensors);
    const auto& lastRun = monteCarloResults.back();

    vector<Target> X = lastRun.X;
    vector<vector<MeasurementRadar>> Z = lastRun.Z;

    vector<TrackOut> track_out;
    vector<pair<double, double>> Z_all;

    for (const auto& trackSet : lastRun.track_list_update) {
        TrackOut tr;
        for (const auto& info : trackSet.track_info) {
            if (info.x.size() >= 3 && !info.x.isZero(1e-3)) {
                double x = info.x(0);
                double y = info.x(2);
                if (!std::isnan(x) && !std::isnan(y)) {
                    tr.pos.emplace_back(x, y);
                }
            }
        }
        if (tr.pos.size() >= 2) track_out.push_back(tr);
    }

    for (const auto& scan : Z) {
        for (const auto& meas : scan) {
            for (const auto& detection : meas.detections) {
                if (detection.size() < 2) continue;
                Z_all.emplace_back(detection[0], detection[1]);
            }
        }
    }

    // Debug print
    cout << "\n========== Range-Azimuth Measurements ==========" << endl;
    for (size_t i = 0; i < Z_all.size(); ++i) {
        cout << "Measurement " << i + 1 << ": Range = " << Z_all[i].first
             << " , Azimuth = " << Z_all[i].second << endl;
    }
    cout << "===============================================\n" << endl;

    // Summary
    cout << " Writing track data: " << track_out.size() << " tracks\n";
    cout << " Writing true state data: " << X.size() << " targets\n";
    int meas_count = 0;
    for (const auto& scan : Z) for (const auto& m : scan) meas_count += m.detections.size();
    cout << " Writing measurement data: " << meas_count << " detections\n";
    cout << " Total range-azimuth pairs: " << Z_all.size() << endl;

    generateGnuplotData(track_out, X, Z);
    cout << " Radar tracking and plotting complete!" << endl;

    return 0;
}