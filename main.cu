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
#include <tuple>
#include <limits>
#include <cub/cub.cuh>
#include <cstdio>

#include <thrust/host_vector.h>


#define N 1000
#define TPB1 16

const int NOISE = -2;
const int NOT_CLASSIFIED = -1;

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(err); \
    }
#define CUDA_CHECK(call)                                                \
    do {                                                               \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                       \
        }                                                              \
    } while (0)


 ///vecchia versione
 struct KDNodeCuda {
    float x, y; // Coordinate del nodo
    int idx;    // Indice del nodo
    int idxLeft;
    int idxRight;
    KDNodeCuda* left;
    KDNodeCuda* right;

    KDNodeCuda(float x_val, float y_val, int index)
            : x(x_val), y(y_val), idx(index), idxLeft(-1), idxRight(-1), left(nullptr), right(nullptr) {}
};

class KDTreeCuda {
private:
    KDNodeCuda* root;

    KDNodeCuda* insert(KDNodeCuda* node, float x, float y, int index, int depth, int &c) {
        if (!node) return new KDNodeCuda(x, y, index);

        int axis = depth % 2;

        if ((axis == 0 && x < node->x) || (axis == 1 && y < node->y)) {
            if (!node->left) {
                // Se il figlio sinistro è vuoto, salva l'indice prima di inserire
                //node->idxLeft = c + 1;  // `c + 1` rappresenta il prossimo nodo a essere visitato
            }
            node->left = insert(node->left, x, y, index, depth + 1, ++c);
        } else {
            if (!node->right) {
                // Se il figlio destro è vuoto, salva l'indice prima di inserire
                //node->idxRight = c + 1;  // `c + 1` rappresenta il prossimo nodo a essere visitato
            }
            node->right = insert(node->right, x, y, index, depth + 1, ++c);
        }

        return node;
    }
    void serializeTree(KDNodeCuda* node, std::vector<KDNodeCuda>& array, int& counter) {
        if (!node) return;

        // Salva l'indice corrente
        int currentIdx = counter;
        counter++;

        // Aggiungi il nodo all'array
        array.push_back(*node);

        // Gestisci il figlio sinistro
        if (node->left) {
            node->idxLeft = counter;  // L'indice del figlio sinistro sarà il valore attuale di `counter`
            serializeTree(node->left, array, counter);
        } else {
            node->idxLeft = -1;  // Nessun figlio sinistro
        }

        // Gestisci il figlio destro
        if (node->right) {
            node->idxRight = counter;  // L'indice del figlio destro sarà il valore attuale di `counter`
            serializeTree(node->right, array, counter);
        } else {
            node->idxRight = -1;  // Nessun figlio destro
        }

        // Aggiorna il nodo corrente nell'array con i valori corretti di idxLeft e idxRight
        array[currentIdx] = *node;
    }

public:
    KDTreeCuda() : root(nullptr) {}

    void insert(float x, float y, int index) {
        int c = 0;
        root = insert(root, x, y, index, 0, c);
    }

    std::vector<KDNodeCuda> toArray() {
        std::vector<KDNodeCuda> array;
        int c = 0;
        serializeTree(root, array, c);
        return array;
    }

    // Funzione per leggere i punti dal file e popolare il KD-Tree
    void readPoints2(const std::string& dir, float *h_xval, float *h_yval, int* h_clusterval, unsigned int* h_nodeDegs, unsigned int*h_neighborStartIndices, unsigned int*h_totalEdges) {
        std::ifstream infile(dir);
        if (!infile.is_open()) {
            std::cout << "Error: Failed to open file." << std::endl;
            return;
        }

        std::string line;
        float x, y;
        int idx = 0;

        while (std::getline(infile, line)) {

            std::istringstream iss(line);
            if (!(iss >> x >> y)) { // Leggi x e y dalla riga
                std::cout << "Error reading line: " << line << std::endl;
                continue; // Salta la riga se non riesci a leggere
            }
            h_xval[idx] = x;
            h_yval[idx] = y;
            //insert(x, y, idx); // Inserisci il punto nel KD-Tree
            h_clusterval[idx] = NOT_CLASSIFIED; // inizializza il vettore clusters a -1
            h_nodeDegs[idx] = 0;    // inizializza vettore contatore vicini a -1
            h_neighborStartIndices[idx] = 0;
            idx++;

        }
        h_totalEdges = 0;
        std::cout << "\nDataset fetched!" << std::endl;
        std::cout << "Punti totali letti: " << idx << std::endl;
    }
};


// Funzione per leggere i punti dal file e popolare il KD-Tree
void readPoints(const std::string& dir, float *h_xval, float *h_yval, int* h_clusterval, unsigned int* h_nodeDegs, unsigned int*h_neighborStartIndices, unsigned int*h_totalEdges) {
    std::ifstream infile(dir);
    if (!infile.is_open()) {
        std::cout << "Error: Failed to open file." << std::endl;
        return;
    }

    std::string line;
    float x, y;
    int idx = 0;

    while (std::getline(infile, line)) {

        std::istringstream iss(line);
        if (!(iss >> x >> y)) { // Leggi x e y dalla riga
            std::cout << "Error reading line: " << line << std::endl;
            continue; // Salta la riga se non riesci a leggere
        }
        h_xval[idx] = x;
        h_yval[idx] = y;
        //h_kdTree.insert(x, y, idx); // Inserisci il punto nel KD-Tree
        h_clusterval[idx] = NOT_CLASSIFIED; // inizializza il vettore clusters a -1
        h_nodeDegs[idx] = 0;    // inizializza vettore contatore vicini a -1
        h_neighborStartIndices[idx] = 0;
        idx++;

    }
    h_totalEdges = nullptr;
    std::cout << "\nDataset fetched!" << std::endl;
    std::cout << "Punti totali letti: " << idx << std::endl;
}

__global__ void makeGraphStep1Kernel(float* d_xval, float* d_yval, KDNodeCuda* kdTree, unsigned int* d_nodeDegs, int numPoints,
                                     float eps, int minPoints, unsigned int* d_totalEdges, unsigned int* d_neighborStartIndices) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= numPoints) return;

    int localStack[N];  // Stack privato per ogni thread
    int stackPtr;

    while (idx < numPoints) {
        unsigned int neighbors = 0; // Numero di vicini trovati per il punto corrente
        unsigned int depht = 0;
        stackPtr = -1;              // Reset dello stack
        localStack[++stackPtr] = 0; // Nodo radice nella posizione iniziale dello stack

        while (stackPtr >= 0) {
            int currentNode = localStack[stackPtr--]; // Pop dallo stack
            if (currentNode < 0 || currentNode >= numPoints) {
                continue;
            }

            KDNodeCuda node = kdTree[currentNode];   // Nodo attuale

            // Calcolo della distanza quadrata
            float distSquared = (node.x - d_xval[idx]) * (node.x - d_xval[idx]) +
                                (node.y - d_yval[idx]) * (node.y - d_yval[idx]);

            // Se il nodo è entro il raggio epsilon, incrementa il conteggio
            if (distSquared <= eps * eps) {
                neighbors++; // Contatore vicini
            }

            int axis = depht  % 2;

            // Determina il sottoalbero vicino e lontano
            int nearNode;
            int farNode;

            if (axis == 0) {
                nearNode = (d_xval[idx] < node.x) ? node.idxLeft : node.idxRight;
                farNode = (nearNode == node.idxLeft) ? node.idxRight : node.idxLeft;

            } else {
                nearNode = (d_yval[idx] < node.y) ? node.idxLeft : node.idxRight;
                farNode = (nearNode == node.idxLeft) ? node.idxRight : node.idxLeft;
            }

            // Aggiungi il sottoalbero vicino allo stack
            if (nearNode >= 0 && nearNode < numPoints && stackPtr < N - 1) {
                if(nearNode != -1) {
                    localStack[++stackPtr] = nearNode;
                    depht++;
                }
            }

            // Controlla se è necessario visitare il sottoalbero lontano
            float axisDist = (axis == 0) ? (d_xval[idx] - node.x) : (d_yval[idx] - node.y);
            if (axisDist * axisDist <= (eps * eps) && farNode >= 0 && farNode < numPoints && stackPtr < N - 1) {
                if(farNode != -1) {
                    localStack[++stackPtr] = farNode;
                    depht++;
                }
            }
        }
        atomicAdd(&d_nodeDegs[idx], neighbors);
        //printf("Thread %d, idx %d: Total neighbors = %d\n", threadIdx.x, idx, d_nodeDegs[idx]);
        //atomicAdd(&d_nodeDegs[idx], neighbors);
        atomicAdd(d_totalEdges, neighbors);
        for (int i = idx + 1; i < numPoints; i++) {
            atomicAdd(&(d_neighborStartIndices[i]), neighbors);  //aggiorno  startIdx di ogni nodo
        }
        idx += blockDim.x * gridDim.x; // Passa al prossimo punto
    }
}


 // versione base
__global__ void makeGraphStep2KernelA(float* d_xval, float* d_yval , const unsigned int* __restrict__ d_neighborStartIndices,
                                     unsigned int* adjList, int numPoints, float eps){

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    while (idx < numPoints) {
        int startIdx = d_neighborStartIndices[idx];
        int countNeighbors = 0;

        for (unsigned int i = 0; i < numPoints; i++) {
            if (idx == i) continue;

            // Calcola la distanza quadrata
            float distSquared = (d_xval[i] - d_xval[idx]) * (d_xval[i] - d_xval[idx]) +
                                (d_yval[i] - d_yval[idx]) * (d_yval[i] - d_yval[idx]);

            // Se la distanza è entro eps, inserisci indice nel vettore di adiacenza
            if (distSquared <= eps * eps) {
                unsigned int curr = startIdx + countNeighbors;
                adjList[curr] = i;
                countNeighbors++;
            }
        }
        idx += blockDim.x * gridDim.x;
    }
}



// * FINE VECCHIA VERSIONE STEP 2

__global__ void makeGraphStep2Kernel(float* d_xval, float* d_yval , const unsigned int* __restrict__ d_neighborStartIndices,
                                     unsigned int* adjList, int numPoints, float eps, KDNodeCuda* kdTree) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= numPoints) return;

    int localStack[N];  // Stack privato per ogni thread
    int stackPtr;

    while (idx < numPoints) {
        //unsigned int neighbors = 0; // Numero di vicini trovati per il punto corrente
        unsigned int depht = 0; // Numero di vicini trovati per il punto corrente
        stackPtr = -1;              // Reset dello stack
        localStack[++stackPtr] = 0; // Nodo radice nella posizione iniziale dello stack
        int startIdx = d_neighborStartIndices[idx];
        int countNeighbors = 0;
        while (stackPtr >= 0) {
            int currentNode = localStack[stackPtr--]; // Pop dallo stack
            if (currentNode < 0 || currentNode >= numPoints) {
                continue;
            }

            KDNodeCuda node = kdTree[currentNode];   // Nodo attuale

            // Calcolo della distanza quadrata
            float distSquared = (node.x - d_xval[idx]) * (node.x - d_xval[idx]) +
                                (node.y - d_yval[idx]) * (node.y - d_yval[idx]);

            // Se il nodo è entro il raggio epsilon, incrementa il conteggio
            if (distSquared <= eps * eps) {
                unsigned int curr = startIdx + countNeighbors;
                adjList[curr] = node.idx;
                countNeighbors++;
            }

            int axis = depht % 2;

            // Determina il sottoalbero vicino e lontano
            int nearNode;
            int farNode;

            if (axis == 0) {
                nearNode = (d_xval[idx] < node.x) ? node.idxLeft : node.idxRight;
                farNode = (nearNode == node.idxLeft) ? node.idxRight : node.idxLeft;

            } else {
                nearNode = (d_yval[idx] < node.y) ? node.idxLeft : node.idxRight;
                farNode = (nearNode == node.idxLeft) ? node.idxRight : node.idxLeft;
            }

            // Aggiungi il sottoalbero vicino allo stack
            if (nearNode >= 0 && nearNode < numPoints && stackPtr < N - 1) {
                if(nearNode != -1) {
                    localStack[++stackPtr] = nearNode;
                    depht++;
                }
            }

            // Controlla se è necessario visitare il sottoalbero lontano
            float axisDist = (axis == 0) ? (d_xval[idx] - node.x) : (d_yval[idx] - node.y);
            if (axisDist * axisDist <= (eps * eps) && farNode >= 0 && farNode < numPoints && stackPtr < N - 1) {
                if(farNode != -1) {
                    localStack[++stackPtr] = farNode;
                    depht++;
                }
            }
            //depht++;

        }
        idx += blockDim.x * gridDim.x;

    }
}


__global__ void count(const float*d_xval, const float*d_yval , unsigned int* d_nodeDegs, int numPoints,
                      float eps, int minPoints, unsigned int*d_totalEdges, unsigned int*d_neighborStartIndices) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < numPoints) {
        float epsSquared = eps * eps;
        unsigned int num = 0;
        for (int i = 0; i < numPoints; i++) {
            if (idx == i) continue;
            // Calcola la distanza quadrata
            float distSquared = ((d_xval[i] - d_xval[idx]) * (d_xval[i] - d_xval[idx])) +
                                ((d_yval[i] - d_yval[idx]) * (d_yval[i] - d_yval[idx]));

            // Se la distanza è entro eps, incrementa contatore vicini
            if (distSquared <= epsSquared) {
                num++;
            }
        }
        d_nodeDegs[idx] = num;
        atomicAdd(d_totalEdges, num);   //aggiorno contatore archi totale
        idx += blockDim.x * gridDim.x;

    }
}

__global__ void BFS(int total_points, int minPts, unsigned int *d_visited, unsigned int *d_frontier, unsigned int *d_frontierCounter, const unsigned int *d_nodeDegs, const unsigned int *d_adjList, const unsigned int *d_neighborStartIndices, int*d_clusterVal, int clusterId) {

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int numFrontier = 0;
    while (idx < total_points) {
        if (d_frontier[idx] == 1) {
            atomicExch(&d_frontier[idx], 0);
            atomicSub(d_frontierCounter, 1);
            atomicExch(&d_visited[idx], 1);
            if (d_clusterVal[idx] == NOT_CLASSIFIED || d_clusterVal[idx] == NOISE) {
                atomicExch(&d_clusterVal[idx], clusterId);
            }
            // Stop BFS se idx non è Core.
            if (d_nodeDegs[idx] >= minPts) {
                unsigned int idx_start = d_neighborStartIndices[idx];
                for (unsigned int i = 0; i < d_nodeDegs[idx]; i++) {
                    unsigned int v = d_adjList[idx_start + i];
                    if (d_visited[v] == 0) {
                        atomicExch(&d_frontier[v], 1);
                        atomicAdd(d_frontierCounter, 1);
                    }
                }
            }
        }
        idx += blockDim.x * gridDim.x;
    }
}
__global__ void clusterPointsSum(unsigned int total_points, float* d_xval, float* d_yval, int* d_clusterVal, float* d_clusterSumX, float* d_clusterSumY, int* d_clusterSize){

    //indice del thread a livello grid
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while(idx<total_points) {
        int clusterId = d_clusterVal[idx];
        if(clusterId >= 0) {
            //sommo tutti i punti appartenenti ai clusters
            atomicAdd(&(d_clusterSumX[clusterId]), d_xval[idx]);
            atomicAdd(&(d_clusterSumY[clusterId]), d_yval[idx]);
            atomicAdd(&(d_clusterSize[clusterId]), 1);
        }
        idx += blockDim.x * gridDim.x;
    }
}
//Definizione classe
class ParallelDBSCAN {
private:
    float eps;
    int minPts;
    int n;
    std::string input_dir;
    std::string output_dir;
    vector<vector<int>> adjPoints;
    int TPB;

public:
    ParallelDBSCAN(std::string output_dir, std::string input_dir, int n , float eps, int minPts, int TPB);
    void run();
    float getDistance(int idx1, int idx2);

    bool isCoreObject(int idx);
    __host__ void bfs(int total_points, int idx, int clusterId, int* h_clusterVal, unsigned int*d_adjList,  unsigned int *d_nodeDegs, unsigned int*d_neighborStartIndices, unsigned int* h_totalEdges);
    ~ParallelDBSCAN() {
        // rilascio tutte le risorse allocate sulla gpu
        cudaDeviceReset();
    }
};
///
///Implementazione Costruttore ParallelDBSCAN
///
ParallelDBSCAN::ParallelDBSCAN(std::string output_dir, std::string input_dir, int n, float eps, int minPts, int TPB){//:points(input_dir)  {
    this->input_dir = input_dir;
    this->output_dir = output_dir;
    this->adjPoints.resize(n);
    this->n = n;
    this->eps = eps;
    this-> minPts = minPts;
    this->TPB = TPB;
};

__host__ void ParallelDBSCAN::bfs(int total_points, int idx, int clusterId, int *h_clusterVal, unsigned int *d_adjList, unsigned int *d_nodeDegs, unsigned int*d_neighborStartIndices, unsigned int* h_totalEdges) {

    unsigned int*h_visited = (unsigned int*)malloc(total_points*sizeof(unsigned int));
    unsigned int*h_frontier = (unsigned int*)malloc(total_points*sizeof(unsigned int));
    unsigned int*h_nodeDegs = (unsigned int*)malloc(total_points*sizeof(unsigned int));
    unsigned int*h_frontierCounter = (unsigned int*)malloc(sizeof(unsigned int));
    unsigned int*d_visited;
    unsigned int*d_frontier;
    unsigned int*d_frontierCounter;
    unsigned int* d_totalEdges;
    int* d_clusterVal;

    memset(h_frontierCounter, 0, sizeof(unsigned int));
    memset(h_visited, 0, total_points * sizeof(unsigned int));
    memset(h_frontier, 0, total_points * sizeof(unsigned int));
    h_frontier[idx] = 1;

    CUDA_CHECK(cudaMalloc(&d_visited, total_points*sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_frontier, total_points*sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_frontierCounter, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_clusterVal, total_points*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_frontier, h_frontier, total_points * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_visited, h_visited, total_points * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontierCounter, h_frontierCounter, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_nodeDegs, d_nodeDegs, total_points*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(d_clusterVal, h_clusterVal, total_points * sizeof(int), cudaMemcpyHostToDevice));
    int numFrontier = 1;

    while(numFrontier > 0){
        memset(h_frontierCounter, 0, sizeof(unsigned int));
        numFrontier = 0;
        CUDA_CHECK(cudaMemcpy(d_frontierCounter, h_frontierCounter, sizeof(unsigned int), cudaMemcpyHostToDevice));
        BFS<<<(N + TPB - 1) / TPB, TPB>>>(total_points, minPts, d_visited, d_frontier, d_frontierCounter, d_nodeDegs, d_adjList, d_neighborStartIndices, d_clusterVal, clusterId);
        // Sincronizza per attendere il completamento del kernel
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaMemcpy(h_frontier, d_frontier, total_points * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_visited, d_visited, total_points * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_frontierCounter, d_frontierCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        numFrontier = *h_frontierCounter;
    }
    CUDA_CHECK(cudaMemcpy(h_clusterVal, d_clusterVal, total_points * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_visited, d_visited, total_points*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_visited));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_clusterVal));
    CUDA_CHECK(cudaFree(d_frontierCounter));
    free(h_visited);
    free(h_frontier);
}

void dfs(int idx, int clusterId, int *h_clusterVal, int minPts, float* h_xval, float* h_yval,unsigned int*  h_adjList, unsigned int* h_totalEdges, unsigned int* h_nodeDegs, unsigned int* h_neighborStartIndices){
    h_clusterVal[idx] = clusterId;
    if(h_nodeDegs[idx] < minPts) return;
    std::vector<int> stack;  // Utilizza un vettore come stack
    stack.push_back(idx);     // Inizializza lo stack con il punto corrente

    while (!stack.empty()) {
        int current = stack.back(); // Ottieni l'elemento in cima allo stack
        stack.pop_back();            // Rimuovi l'elemento dallo stack
        unsigned int start = h_neighborStartIndices[current];
        // Ottieni i vicini del punto corrente
        for(int j = 0; j < h_nodeDegs[current];j++){
            int adj = h_adjList[start + j];
            if(h_clusterVal[adj] == NOT_CLASSIFIED || h_clusterVal[adj] == NOISE){
                h_clusterVal[adj] = clusterId;
                if(h_nodeDegs[adj] >= minPts){
                    stack.push_back(adj);
                }
            }
        }
    }
}

void ParallelDBSCAN::run() {
    //alloco memoria Host
    int total_points = n;
    std::cout << n << std::endl;
    float *h_xval = (float *) malloc(N * sizeof(float));
    float *h_yval = (float *) malloc(N * sizeof(float));
    int *h_clusterVal = (int *) malloc(N * sizeof(int));
    unsigned int *h_nodeDegs = (unsigned int *) malloc(N * sizeof(unsigned int));
    unsigned int *h_totalEdges = (unsigned int *) malloc(sizeof(unsigned int));
    unsigned int *h_neighborStartIndices = (unsigned int *) malloc(N * sizeof(unsigned int));

    //alloco memoria Device
    float *d_xval;
    float *d_yval;
    int *d_clusterVal;  //salva labels clusters
    unsigned int *d_nodeDegs;      //contatore vicini
    unsigned int *d_adjList;   //vettore-matrice per adiacenza
    unsigned int *d_totalEdges;         //Totale archi adjList


    unsigned int *d_neighborStartIndices;


    CUDA_CHECK(cudaMalloc(&d_xval, total_points * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_yval, total_points * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nodeDegs, total_points * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_neighborStartIndices, total_points * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_totalEdges, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_clusterVal, total_points * sizeof(int)));

    readPoints(input_dir, h_xval, h_yval, h_clusterVal, h_nodeDegs, h_neighborStartIndices, h_totalEdges);

    CUDA_CHECK(cudaMemcpy(d_xval, h_xval, total_points * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_yval, h_yval, total_points * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighborStartIndices, h_neighborStartIndices, total_points * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    //CUDA_CHECK(cudaMemcpy(d_kdTree, array.data(), sizeof(KDNodeCuda) * total_points, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_totalEdges,0,sizeof(unsigned int)));
    //CUDA_CHECK(cudaMemcpy(d_kdTree, h_kdTree, sizeof(KDNodeCuda) * total_points, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodeDegs, h_nodeDegs, total_points * sizeof(unsigned int), cudaMemcpyHostToDevice));

    //Tree
    //Kernel per definire grado di ogni nodo e num totale archi
    count<<<(N + TPB - 1) / TPB, TPB>>>(d_xval, d_yval,
                                        d_nodeDegs, total_points,
                                        eps, minPts, d_totalEdges, d_neighborStartIndices);

    CUDA_CHECK(cudaMemcpy(h_totalEdges, d_totalEdges, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h_nodeDegs, d_nodeDegs, total_points * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::cout << "Lista di adiacenza creata, archi totali: " << *h_totalEdges << std::endl;

    //alloco vettore di adiacenza h e d
    auto *h_adjList = (unsigned int *) malloc(*h_totalEdges * sizeof(unsigned int));
    CUDA_CHECK(cudaMalloc(&d_adjList, *h_totalEdges * sizeof(unsigned int)));

    ///
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // 1. Ottieni la quantità di memoria temporanea richiesta
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_nodeDegs, d_neighborStartIndices, total_points);
    // 2. Alloca la memoria temporanea
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // 3. Esegui il calcolo della prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_nodeDegs, d_neighborStartIndices, total_points);
    cudaFree(d_temp_storage);
    ///

    makeGraphStep2KernelA<<<(N + TPB - 1) / TPB, TPB>>>(d_xval, d_yval, d_neighborStartIndices,
                                                       d_adjList, total_points, eps);//d_kdTree
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemcpy(h_neighborStartIndices, d_neighborStartIndices, total_points * sizeof(unsigned int),
                                             cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_adjList, d_adjList, *h_totalEdges * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    auto start = std::chrono::high_resolution_clock::now();
    ///Identify Cluster
    int clusterId = 0;
    int count = 0;
    for (int i = 0; i < total_points; i++) {
        if (h_clusterVal[i] != NOT_CLASSIFIED) continue;
        if (h_nodeDegs[i] > minPts) { //se CORE POINT
            bfs(total_points, i, clusterId, h_clusterVal, d_adjList, d_nodeDegs, d_neighborStartIndices, d_totalEdges);
            clusterId++;
        } else {
            h_clusterVal[i] = NOISE;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Tempo di esecuzione BFS: " << duration.count() << " millisecondi" << std::endl;
    cudaFree(d_adjList);
    cudaFree(d_totalEdges);
    free(h_adjList);
    free(h_totalEdges);
    h_totalEdges = nullptr;
    h_adjList = nullptr;
    float *h_clusterSumX = (float *) malloc(clusterId *sizeof(float));
    float *h_clusterSumY = (float *) malloc(clusterId * sizeof(float));
    float *d_clusterSumX;
    float *d_clusterSumY;
    CUDA_CHECK(cudaMalloc(&d_clusterSumX, clusterId * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_clusterSumY, clusterId * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_clusterSumX, 0.0, clusterId * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_clusterSumY, 0.0, clusterId * sizeof(float)));
    int *h_clusterSize = (int *) malloc(clusterId *sizeof(int));
    int *d_clusterSize;
    CUDA_CHECK(cudaMalloc(&d_clusterSize, clusterId * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_clusterSize, 0, clusterId * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clusterVal, total_points * sizeof(int)));
    cudaMemcpy(d_clusterVal, h_clusterVal, total_points * sizeof(int), cudaMemcpyHostToDevice);
    float *h_centroidX = (float *) malloc(clusterId *sizeof(float));
    float *h_centroidY = (float *) malloc(clusterId * sizeof(float));

    clusterPointsSum<<<(N + TPB - 1) / TPB, TPB>>>(total_points, d_xval, d_yval, d_clusterVal, d_clusterSumX, d_clusterSumY,
                                                   d_clusterSize);

    //copia variabili aggiornate da device -> host

    CUDA_CHECK(cudaMemcpy(h_clusterSumY, d_clusterSumY, clusterId * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_clusterSumX, d_clusterSumX, clusterId * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_clusterSize, d_clusterSize, clusterId * sizeof(int), cudaMemcpyDeviceToHost));

    //Ricalcolo nuovi centroidi per ogni cluster
    for (int i = 0; i < clusterId; i++) {
        h_centroidX[i] = h_clusterSumX[i] / h_clusterSize[i];
        h_centroidY[i] = h_clusterSumY[i] / h_clusterSize[i];
        std::cout<<h_centroidX[i]<<" "<<h_centroidY[i]<<std::endl;
    }
    //scrittura centroidi
    std::ofstream outfile;
    outfile.open(output_dir + "/" + "clusters.txt");
    if (outfile.is_open()) {
        for (int i = 0; i < clusterId; i++) {
            std::cout << "Cluster " << i << " centroid : ";

            std::cout << h_centroidX[i] << " "<< h_centroidY[i] << std::endl;    // Output console
            outfile << h_centroidX[i] << " " << h_centroidY[i]; // Output file
            outfile << std::endl;
        }
        std::cout << std::endl;
        outfile.close();
    } else {
        std::cout << "Error: Unable to write to clusters.txt";
    }


    std::cout << output_dir + "/" + "clustering.txt" << std::endl;
    outfile.open(output_dir + "/" + "clustering.txt");
    for (int i = 0; i < n; i++) {
        if (outfile.is_open()) {
            outfile << h_xval[i] << " " << h_yval[i] << " "
                    << h_clusterVal[i];// Output to file
        } else {
            std::cerr << "Error: Failed to open file " << output_dir + "/" + "clustering.txt" << std::endl;
        }

        outfile << std::endl;
    }
    //rilascio risorse
    cudaFree(d_xval);
    cudaFree(d_yval);
    cudaFree(d_clusterVal);


    cudaFree(d_nodeDegs);
    cudaFree(d_neighborStartIndices);
    cudaFree(d_clusterSumX);
    cudaFree(d_clusterSumY);
    cudaFree(d_clusterSize);


    free(h_xval);
    free(h_yval);
    free(h_clusterVal);
    free(h_nodeDegs);
    free(h_neighborStartIndices);
    free(h_centroidX);
    free(h_centroidY);
    free(h_clusterSize);
    free(h_clusterSumX);
    free(h_clusterSumY);
    cudaDeviceReset();

}


float runSequential(int n, float eps, int minPts, std::string inputFileName, std::string output_dir, int numTest) {

    float mediaS;
    float sum;

    for (int i = 0; i < numTest; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        SequentialDBSCAN dbScan(output_dir, inputFileName, n, eps, minPts);
        dbScan.run();
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Tempo di esecuzione Sequenziale: " << duration.count() << " millisecondi" << std::endl;
        sum += static_cast<double>(duration.count());
        std::cout << "<<------------------------------>>" << std::endl;

    }
    mediaS = static_cast<double>(sum) / numTest;
    return mediaS;
}
float runParallel(int n, float eps, int minPts, const std::string inputFileName, std::string output_dir, int numTest, int TPB){
    float mediaP;
    float sum;
    for(int i  = 0; i < numTest; i++ ) {

        auto start = std::chrono::high_resolution_clock::now();
        ParallelDBSCAN parallelDbscan(output_dir, inputFileName, n, eps, minPts, TPB);

        parallelDbscan.run();
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Tempo di esecuzione Parallela TPB: " << TPB << " : " << duration.count() << " millisecondi"
                  << std::endl;
        sum += static_cast<float>(duration.count());
        std::cout << "<<------------------------------>>" << std::endl;


    }
    mediaP = static_cast<float>(sum)/numTest;
    return mediaP;

}

int main() {

    string inputFileName("../cmake-build-debug-remote-host/inputG.txt");
    std::string output_dir= "../cmake-build-debug-remote-host/cluster_details";
    int n = N;
    double eps = 10;
    int minPts = 250;
    std::vector<int> TPB = {8,16,128,512};

    int numTest = 3;
    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    for(int j = 0; j< TPB.size();j++) {
        auto mediaP = runParallel(n, eps, minPts, inputFileName, output_dir, numTest, TPB[j]);
        auto mediaS =runSequential(n, eps, minPts, inputFileName, output_dir, numTest);
        std::cout << "Media esecuzione Sequenziale : " << mediaS << std::endl;
        std::cout << "Media esecuzione Parallela TPB: "<<TPB[j]<<" :" << mediaP << std::endl;
        //double mediaS1 = 123048;
        double speedup = static_cast<double>(mediaS) / static_cast<double>(mediaP);
        std::cout << "Speedup: " << speedup << std::endl;
    }
    return 0;
}




/*
    KDNode* h_kdTree = (KDNode *) malloc(N * sizeof(KDNode));
   KDTreeCuda h_kdTree;

    //KDNodeCuda *h_kdTree = (KDNodeCuda *) malloc(N * sizeof(KDNodeCuda));
    //unsigned int *h_adjList;

    //std::vector<KDNodeCuda> kdTree
     //KDNodeCuda *d_kdTree;   //Array KD Tree
    */
    //tree.readPoints(input_dir, h_xval, h_yval, h_clusterVal, h_nodeDegs, h_neighborStartIndices, h_totalEdges);
    //std::vector<KDNodeCuda> array = h_kdTree.toArray();  //serializza KD Tree in Array
    //CUDA_CHECK(cudaMalloc(&d_kdTree, sizeof(KDNodeCuda) * total_points));

     //chiamata al Kernel per riempire l'array di adiacenza
    //makeGraphStep2Kernel<<<(N + TPB - 1) / TPB, TPB>>>(d_xval, d_yval, d_neighborStartIndices,
    //                                                   d_adjList, total_points, eps, d_kdTree);//d_kdTree


/* Tree
    makeGraphStep1Kernel<<<(N + TPB - 1) / TPB, TPB>>>(d_xval, d_yval, d_kdTree,
                                                       d_nodeDegs, total_points,
                                                       eps, minPts, d_totalEdges, d_neighborStartIndices);

    CUDA_CHECK(cudaMemcpy(h_totalEdges, d_totalEdges, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h_nodeDegs, d_nodeDegs, total_points * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_neighborStartIndices, d_neighborStartIndices, total_points * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    */