#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as mpathces
from sklearn import datasets
iris = datasets.load_iris()
x = iris.data[:, 0]
y = iris.data[:, 1]
species = iris.target

x_min, x_max = x.min() - .5, x.max() + .5
y_min, y_max = y.min() - .5, y.max() + .5

plt.figure()
plt.title('Iris Dataset - Classfication By Sepal Size')
plt.scatter(x, y, c=species)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()