//
// Created by Giacomo Magistrato on 10/10/24.
//

#ifndef DBSCAN_PARALLEL_SEQUENTIALDBSCAN_H
#define DBSCAN_PARALLEL_SEQUENTIALDBSCAN_H

#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include "Points.h"

class SequentialDBSCAN {
private:
    float eps;
    int minPts;
    int n;
    std::string input_dir;
    std::string output_dir;
    Points points;
    int numPoints;
    vector<vector<int>> adjPoints;
    int clusterIdx;

public:
    SequentialDBSCAN(std::string output_dir, std::string input_dir, int n , float eps, int minPts);
    void run();
    float getDistance(int idx1, int idx2);
    ~SequentialDBSCAN() {
        adjPoints.clear();  // Libera la memoria associata ai vettori nidificati.
    }
    bool isCoreObject(int idx);

    void dfs(int now, int c);
    void checkNearPoints(int size);
};


#endif //DBSCAN_PARALLEL_SEQUENTIALDBSCAN_H
