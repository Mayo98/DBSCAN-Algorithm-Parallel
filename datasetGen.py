#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:17:13 2024

@author: giacomomagistrato
"""
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=100000, noise=0.2, random_state=1)
X_scaled = X * 60
with open('inputD.txt', 'w') as f:
    for i in range(X.shape[0]):
        f.write(f"{X_scaled[i, 0]} {X_scaled[i, 1]}\n")  # Formato: x y

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
plt.title('Moon-shaped dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Classe')
plt.grid(True)
plt.show()