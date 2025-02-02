//
// Created by Giacomo Magistrato on 10/10/24.
//

#include "SequentialDBSCAN.h"
#include "Points.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <algorithm>

const int NOISE = -2;
const int NOT_CLASSIFIED = -1;

SequentialDBSCAN::SequentialDBSCAN(std::string output_dir, std::string input_dir, int n, float eps, int minPts):points(input_dir)  {
    this->input_dir = input_dir;
    this->output_dir = output_dir;
    this->numPoints = points.getDimensions();
    this->adjPoints.resize(numPoints);
    this->clusterIdx = 0;
    this->n = n;
    this->eps = eps;
    this-> minPts = minPts;
}

float SequentialDBSCAN::getDistance(int idx1, int idx2){

    //return the Euclidean Distance
    float result = pow(points.getXval(idx1)-points.getXval(idx2),2) + pow(points.getYval(idx1)-points.getYval(idx2),2);
    return result;
}


void SequentialDBSCAN::checkNearPoints(int size) {
    float epsSquared = eps * eps; // Calcola il quadrato di epsilon una sola volta
    for (int i = 0; i < size; i++) {
        float x1 = points.getXval(i);
        float y1 = points.getYval(i);
        std::vector<std::tuple<float, float, int>> nearbyPoints = points.findNearby(x1,y1, epsSquared);


        for (const auto& point : nearbyPoints) {
            float x2 = std::get<0>(point); // Accesso alla coordinata x
            float y2 = std::get<1>(point); // Accesso alla coordinata y
            int index = std::get<2>(point); // Accesso all'indice

            // Calcola la distanza quadrata
            float distSquared = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);

            if (distSquared <= epsSquared) {
                points.addNeighbor(i);
                adjPoints[i].push_back(index); // Aggiungi indice alla lista dei vicini
            }
        }
    }
}
bool SequentialDBSCAN::isCoreObject(int idx) {
    //check if point i'th has >= minPts neighbors
    int num = points.getNumPoints(idx);
    if(num >= minPts)
    {
        return true;
    }else return false;
}



void SequentialDBSCAN::dfs (int now, int c) {
    // Segna il punto attuale come parte del cluster c
    points.setCluster(now, c);

    // Se il punto non è un oggetto core, termina la funzione
    if (!isCoreObject(now)) return;

    std::vector<int> stack;  // Utilizza un vettore come stack
    stack.push_back(now);     // Inizializza lo stack con il punto corrente

    while (!stack.empty()) {
        int current = stack.back(); // Ottieni l'elemento in cima allo stack
        stack.pop_back();            // Rimuovi l'elemento dallo stack

        // Ottieni i vicini del punto corrente
        for (int next: adjPoints[current]) {
            // Controlla se il punto non è già classificato
            if (points.getCluster(next) == NOT_CLASSIFIED || points.getCluster(next) == NOISE) {
                points.setCluster(next, c);  // Classifica il vicino
                // Solo se il vicino è un oggetto core, lo aggiungi allo stack
                if (isCoreObject(next)) {
                    stack.push_back(next); // Aggiungi il vicino allo stack
                }
            }
        }
    }
}


void SequentialDBSCAN::run() {

    checkNearPoints(numPoints);
    auto start = std::chrono::high_resolution_clock::now();
    clusterIdx = 0;
    for (int i = 0; i < numPoints; i++) {
        if (points.getCluster(i) != NOT_CLASSIFIED) continue;

        if (isCoreObject(i)) {
            dfs(i, clusterIdx);
            clusterIdx++;
        } else {
            points.setCluster(i, NOISE);
        }
    }
    std::cout << "Clusters: " << clusterIdx << std::endl;
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Tempo di esecuzione DFS: " << duration.count() << " millisecondi" << std::endl;
    std::vector<float> clusterSumX;
    std::vector<float> clusterSumY;
    std::vector<float> clusterPointsNum;
    std::vector<float> centroidX;
    std::vector<float> centroidY;
    for (int i = 0; i < clusterIdx; i++) {
        clusterSumX.push_back(0.0);
        clusterSumY.push_back(0.0);
        clusterPointsNum.push_back(0.0);
        centroidX.push_back(0.0);
        centroidY.push_back(0.0);
    }
    for (int i = 0; i < numPoints; i++) {
        float x = points.getXval(i);
        float y = points.getYval(i);
        int clusterId = points.getCluster(i);
        if(clusterId >= 0) {
            clusterPointsNum[clusterId]++;
            clusterSumX[clusterId] += x;
            clusterSumY[clusterId] += y;
        }
        //clusters[clusterId].addPoint();
    }
    for (int i = 0; i < clusterIdx; i++) {
        if(clusterPointsNum[i]> 0) {
            centroidX[i] = clusterSumX[i] / clusterPointsNum[i];
            centroidY[i] = clusterSumY[i] / clusterPointsNum[i];
        }
    }

    std::ofstream outfile;
    std::cout << std::filesystem::current_path().string() << std::endl;
    std::cout << output_dir + "/" + "clustersS.txt" << std::endl;
    outfile.open(output_dir + "/" + "clustersS.txt");
    if (outfile.is_open()) {
        for (int i = 0; i < clusterIdx; i++) {
            //std::cout <<  i << " cluster contiene: "<< clusters[i].getSize() <<std::endl;
            std::cout << "Cluster " << i << " centroid : ";
            std::cout << centroidX[i] << " " << centroidY[i] << std::endl;    // Output console
            outfile << centroidX[i] << " " << centroidY[i]; // Output file
            outfile << std::endl;
        }
        std::cout << std::endl;
        outfile.close();
    } else {
        std::cout << "Error: Unable to write to clusters.txt";
    }

    //std::ofstream outfile;
    std::cout << output_dir + "/" + "clusteringS.txt" << std::endl;
    outfile.open(output_dir + "/" + "clusteringS.txt");
    if (outfile.is_open()) {
        for (int i = 0; i < numPoints; i++) {
            outfile << points.getXval(i) << " " << points.getYval(i) << " "
                    << points.getCluster(i);// Output to file
            outfile << std::endl;
        }
        std::cout << std::endl;
        outfile.close();
    } else {
        std::cerr << "Error: Failed to open file " << output_dir + "/" + "clusteringS.txt" << std::endl;
    }

    clusterSumX.clear();
    clusterSumX.shrink_to_fit();
    clusterSumY.clear();
    clusterSumY.shrink_to_fit();
    clusterPointsNum.clear();
    clusterPointsNum.shrink_to_fit();
    centroidX.clear();
    centroidX.shrink_to_fit();
    centroidY.clear();
    centroidY.shrink_to_fit();
}


