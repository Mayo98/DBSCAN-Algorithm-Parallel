#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:03:33 2023

@author: giacomomagistrato
"""

import numpy as np
import matplotlib.pyplot as plt
import random


# genero un numero di punti casuali per ciascun cluster, se voglio fissare lo stesso numero di punti x cluster uso num_points

def generate_dataset(num_points, cluster_means, cluster_std_devs, plot=False):
    dataset = []
    for mean, std_dev in zip(cluster_means, cluster_std_devs):
        cluster = np.round(np.random.normal(mean, std_dev, (num_points, 2)),2)
        dataset.append(cluster)
        print(cluster)
    dataset = np.concatenate(dataset)


    if plot:
        plt.scatter(dataset[:, 0], dataset[:, 1])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Generated Dataset')
        plt.show()

    return dataset.tolist()

#num_points = random.randint(10, 1000) se voglio un numero di punti casuali

# Esempio di utilizzo con visualizzazione del plot
num_points = 100000#250000  # Numero di punti nel dataset
num_clusters = 6  # Numero di cluster

# Definizione dei cluster
cluster_means = []
cluster_std_devs = []

for _ in range(num_clusters):
    #mean = np.random.uniform(low=-600, high=700, size=2)  # Genera una media casuale per il cluster
    #std_dev = np.random.uniform(low=0.5, high=100, size=2)  # Genera una deviazione standard casuale per il cluster
    #cluster_means.append(mean)
    #cluster_std_devs.append(std_dev)

    mean = np.random.uniform(low=-100, high=200, size=2)
    while any(np.linalg.norm(mean - np.array(cm)) < 100 for cm in cluster_means):
        mean = np.random.uniform(low=-200, high=600, size=2)
    cluster_means.append(mean)

    # Genera una deviazione standard fissa per il cluster
    std_dev = np.array([30, 30])
    cluster_std_devs.append(std_dev)



dataset = generate_dataset(num_points, cluster_means, cluster_std_devs, plot=True)

np.savetxt("./cmake-build-debug/inputG.txt", dataset, delimiter=' ', fmt='%.2f')
i = 0
for elem in dataset:
    i = i+1;
print(i)


