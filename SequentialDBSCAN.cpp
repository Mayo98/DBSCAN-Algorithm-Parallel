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
#include <chrono>
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
void SequentialDBSCAN::checkNearPoints(int size) {
    float epsSquared = eps * eps;
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            if(i==j) continue;
            if(getDistance(i,j) <= epsSquared) {
                points.addNeighbor(i);
                adjPoints[i].push_back(j); //aggiungo indice alla lista dei vicini
            }
        }
    }
}
float SequentialDBSCAN::getDistance(int idx1, int idx2){

    //return the Euclidean Distance
    float result = pow(points.getXval(idx1)-points.getXval(idx2),2) + pow(points.getYval(idx1)-points.getYval(idx2),2);
    return result;
}

void SequentialDBSCAN::checkNearPoints2(int size) {
    float epsSquared = eps * eps; // Calcola il quadrato di epsilon una sola volta

    for (int i = 0; i < size; i++) {
        if(i == size/3)
        {
            std::cout<<"Sono a metà"<<std::endl;
        }
        float x1 = points.getXval(i);
        float y1 = points.getYval(i);

        for (int j = 0; j < size; j++) {
            if (i == j) continue; // Ignora il punto stesso

            float x2 = points.getXval(j);
            float y2 = points.getYval(j);

            // Calcola la distanza quadrata
            float distSquared = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);

            if (distSquared <= epsSquared) {
                points.addNeighbor(i);
                adjPoints[i].push_back(j); // Aggiungi indice alla lista dei vicini
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
    points.setCluster(now, c);
    if(!isCoreObject(now)) return;

    for(auto&next:adjPoints[now]) {
        if(points.getCluster(next) != NOT_CLASSIFIED) continue;
        dfs(next, c);
    }
}

void SequentialDBSCAN::dfs2 (int now, int c) {
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
            if (points.getCluster(next) == NOT_CLASSIFIED) {
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
    auto start = std::chrono::high_resolution_clock::now();
    checkNearPoints2(numPoints);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Tempo di esecuzione checkNearPoints: " << duration.count() << " millisecondi" << std::endl;
    int c = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numPoints; i++) {
        if (points.getCluster(i) != NOT_CLASSIFIED) continue;

        if (isCoreObject(i)) {
            dfs2(i, ++clusterIdx);
        } else {
            points.setCluster(i, NOISE);
        }
        c++;
    }
    end = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Tempo di esecuzione dfs: " << duration.count() << " millisecondi" << std::endl;
    cout<<"c = "<< c<<endl;


    std::ofstream outfile;
    std::cout << output_dir + "/" + "clusteringS.txt" << std::endl;
    outfile.open(output_dir + "/" + "clusteringS.txt");
    for (int i = 0; i < numPoints; i++) {
        if (outfile.is_open()) {
            outfile << points.getXval(i) << " " << points.getYval(i) << " "
                    << points.getCluster(i);// Output to file
        }
        else {
            std::cerr << "Error: Failed to open file " << output_dir + "/" + "clusteringS.txt" << std::endl;
        }

        outfile << std::endl;
/*
    cluster.resize(clusterIdx+1);
    for(int i=0;i<size;i++) {
        if(points[i].cluster != NOISE) {
            cluster[points[i].cluster].push_back(i);
        }
    }*/
    }
}

