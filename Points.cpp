//
// Created by Giacomo Magistrato on 10/10/24.
//

#include "Points.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <algorithm>

using namespace std;


void Points::lineToVec(std::string& dir) {

    std::string tmp = "";
    string line;
    //std::cout << std::filesystem::current_path().string() <<"Ciao" << std::endl;
    ifstream infile("../cmake-build-debug-remote-host/" + dir);
    if (!infile.is_open()) {
        cout << "Error: Failed to open file." << endl;
        return;
    }
    float x,y;
    int idx = 0;
    while (getline(infile, line)) {
        for (int i = 0; i < static_cast<int>(line.length()); i++) {
            if ((48 <= static_cast<int>(line[i]) && static_cast<int>(line[i]) <= 57) || line[i] == '.' ||
                line[i] == '+' || line[i] == '-' || line[i] == 'e') {
                tmp += line[i];

            } else if (!tmp.empty()) {
                xval.push_back(std::stod(tmp));
                x = std::stod(tmp);
                tmp = "";
            }

        }
        if (!tmp.empty()) {
            yval.push_back(std::stod(tmp));
            y = std::stod(tmp);
            tmp = "";
        }
        this->kdTree.insert(x,y,idx++);
        this->clusters.push_back(-1); //inizializzo anche il vettore clusters a -1 NOT_CLASSIFIED
        this->ptsCnt.push_back(0);  // inizializzo vettore contatore vicini a 0
    }
    std::cout<<"dim clusters: "<<clusters.size()<<std::endl;

    cout << "\nDataset fetched!" << endl
         << endl;
    std::cout << "Punti totali letti : " << xval.size() << std::endl;

}
Points::Points(std::string dir): kdTree()
{
    this->lineToVec(dir);
    this->dimensions = xval.size();

}

int Points::getDimensions()
{
    return dimensions;
}

int Points::getCluster(int idx)
{
    return clusters[idx];
}
float Points::getXval(int pos)
{
    return xval[pos];
}
float Points::getYval(int pos)
{
    return yval[pos];
}
int Points::getNumPoints(int idx) {

    return ptsCnt[idx];
}
void Points::addNeighbor(int idx) {
    this->ptsCnt[idx]++;
}
void Points::setCluster(int idx, int val) {
    this->clusters[idx] = val;
}

std::vector<std::tuple<float, float, int>> Points::findNearby(float x, float y, float eps) {
    std::vector<std::tuple<float, float, int>> nearby = kdTree.findNearby(x,y, eps);
    return nearby;
}
