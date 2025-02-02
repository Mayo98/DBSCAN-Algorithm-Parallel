//
// Created by Giacomo Magistrato on 21/10/24.
//

#ifndef DBSCAN_PARALLEL_KDNODE_H
#define DBSCAN_PARALLEL_KDNODE_H


class KDNode {
public:
    float x; // Coordinata x del punto
    float y; // Coordinata y del punto
    int idx;
    KDNode* left; // Sottoalbero sinistro
    KDNode* right; // Sottoalbero destro

    KDNode(float x, float y, int idx) : x(x), y(y),idx(idx), left(nullptr), right(nullptr) {}

};
#endif //DBSCAN_PARALLEL_KDNODE_H
