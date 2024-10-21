#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:02:44 2023

@author: giacomomagistrato
"""


import matplotlib.pyplot as plt
import numpy as np

centroids = {}
centroX = []
centroY = []

#with open('4-clusters.txt','r') as input_file:

points = {}

puntoX = []
puntoY = []
puntoZ = []
with open('cmake-build-debug/cluster_details/clusteringS.txt', 'r') as f:
    lines = f.readlines()
    i = 0
    for line in lines:
        x, y, z = line.strip().split(' ')
        puntoX.append(float(x))
        puntoY.append(float(y))
        puntoZ.append(float(z))
        points[float(x)] = (float(y),float(z))
        i = i + 1
print(i)
print(len(puntoX))
fig, ax = plt.subplots()
x1 =[]
y1 = []
z1 = []
for chiave, valori in points.items():
    y, z = valori
    y1.append(y)
    z1.append(z)
    x1.append(chiave)



#plt.scatter(x1, y1, c = z1,   cmap = "viridis", label = 'points')

plt.scatter(puntoX, puntoY, c = puntoZ,   cmap = "viridis", label = 'points')
plt.scatter(centroX, centroY, c ="red", label = 'centroids')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('k-Means Clustering')
plt.savefig('result.png')
plt.show()


ax.legend()




