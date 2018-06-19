#!/usr/bin/env python3
import matplotlib.pyplot as plt
plt.axis([0,5,0,20])
plt.title('My first plot', fontsize=20, fontname='Times New Roman')
plt.xlabel('Counting', color='gray')
plt.ylabel('Square values', color='gray')
plt.text(1,1.5, 'First')
plt.text(2,4.5, 'Second')
plt.text(3, 9.5, 'Third')
plt.text(4, 16.5, 'Fourth')
plt.text(1.1, 12, r'$y = x^2$', fontsize=20,bbox={'facecolor': 'yellow', 'alpha':0.2})
plt.grid(True)
plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.show()