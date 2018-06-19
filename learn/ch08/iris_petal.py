#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import datasets

iris = datasets.load_iris()
x = iris.data[:, 2] # X-Axis -petal length
y = iris.data[:, 3] #Y-Axis  -petal width
species = iris.target #species

x_min, x_max = x.min() - .5, x.max() + .5
y_min, y_max = y.min() - .5, y.max() + .5

plt.figure()
plt.title('Iris Dataset - Classfication By Petal Sizes', size=14)
plt.scatter(x, y, c=species)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()