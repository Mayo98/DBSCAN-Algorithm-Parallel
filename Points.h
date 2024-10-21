//
// Created by Giacomo Magistrato on 10/10/24.
//

#ifndef DBSCAN_PARALLEL_POINTS_H
#define DBSCAN_PARALLEL_POINTS_H

#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <algorithm>

using namespace std;

class Points {
private:
    int dimensions ;
    //std::vector<float> values;
    void lineToVec(std::string& line);

    std::vector<float>xval;
    std::vector<float>yval;

    std::vector<int>clusters;
    std::vector<int>ptsCnt;
public:
    Points(std::string dir);
    int getDimClusters();
    int getDimensions();
    int getCluster(int idx);
    void setCluster(int idx, int val);
    int getNumPoints(int idx);
    void addNeighbor(int idx);

    float getXval(int pos);
    void initCluster(int dimensions);
    float getYval(int pos);

};

#endif //DBSCAN_PARALLEL_POINTS_H
