//
// Created by Giacomo Magistrato on 21/10/24.
//

#ifndef DBSCAN_PARALLEL_KDTREE_H
#define DBSCAN_PARALLEL_KDTREE_H

#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include "KDNode.h"

class KDTree {
private:

    KDNode *root;

// Funzione ricorsiva per inserire un punto nel KD-Tree
    KDNode *insert(KDNode *node, float x, float y, int index, int depth) {
        if (!node) return new KDNode(x, y, index);

        // Determina l'asse (0 = x, 1 = y) da utilizzare per l'inserimento
        int axis = depth % 2;

        if ((axis == 0 && x < node->x) || (axis == 1 && y < node->y)) {
            node->left = insert(node->left, x, y, index, depth + 1);
        } else {
            node->right = insert(node->right, x, y, index, depth + 1);
        }

        return node;
    }

    void searchNearby(KDNode* node, float targetX, float targetY, float epsSquared, std::vector<std::tuple<float, float, int>>& foundPoints, int depth) {
        if (!node) return;

        // Calcola la distanza quadrata al nodo corrente
        float distSquared = getDistance(node->x, node->y, targetX, targetY);

        // Aggiungi il nodo se la distanza è entro l'eps
        if (distSquared <= epsSquared) {
            foundPoints.emplace_back(node->x, node->y, node->idx);
        }

        // Determina l'asse (0 = x, 1 = y) da utilizzare per la ricerca
        int axis = depth % 2;
        KDNode* nearNode = (axis == 0) ? (targetX < node->x ? node->left : node->right) : (targetY < node->y ? node->left : node->right);
        KDNode* farNode = (nearNode == node->left) ? node->right : node->left;

        // Ricerca nel sottoalbero vicino
        searchNearby(nearNode, targetX, targetY, epsSquared, foundPoints, depth + 1);

        // Controlla se dobbiamo visitare il sottoalbero lontano
        float axisDist = (axis == 0) ? (targetX - node->x) : (targetY - node->y);
        if (axisDist * axisDist <= epsSquared) {
            searchNearby(farNode, targetX, targetY, epsSquared, foundPoints, depth + 1);
        }
    }

    float getDistance(float x1, float y1, float x2, float y2) {
        return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
    }
public:
    KDTree() : root(nullptr) {}

// Metodo per inserire un punto nel KD-Tree usando x, y e index
    void insert(float x, float y, int index) {
        root = insert(root, x, y, index, 0);
    }

// Metodo per trovare punti vicini
    std::vector<std::tuple<float, float, int>> findNearby(float targetX, float targetY, float eps) {
        std::vector<std::tuple<float, float, int>> foundPoints;
        searchNearby(root, targetX, targetY, eps, foundPoints, 0);
        return foundPoints;
    }
};

#endif //DBSCAN_PARALLEL_KDTREE_H
