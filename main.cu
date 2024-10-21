#include <iostream>
#include <cuda_runtime.h>
#include "Points.h"
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <filesystem>
#include "SequentialDBSCAN.h"
#include <algorithm>



int main() {

    string inputFileName("inputG.txt");
    int n = 1000;
    double eps = 20;
    int minPts = 300;


    std::string output_dir= "../cmake-build-debug-remote-host/cluster_details";
    auto start = std::chrono::high_resolution_clock::now();
    SequentialDBSCAN dbScan(output_dir, inputFileName, n, eps, minPts);
    dbScan.run();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Tempo di esecuzione Sequenziale: " << duration.count() << " millisecondi" << std::endl;

    return 0;
}
