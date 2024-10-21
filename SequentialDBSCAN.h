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
    void checkNearPoints(int size);
    std::vector<int>used_pointIds;
    void run();
    float getDistance(int idx1, int idx2);

    bool isCoreObject(int idx);

    void dfs(int now, int c);

    void dfs2(int now, int c);

    void checkNearPoints2(int size);
};


#endif //DBSCAN_PARALLEL_SEQUENTIALDBSCAN_H
