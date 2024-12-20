#https://www.cnblogs.com/pinard/p/6812011.html

import numpy as np
from sklearn.decomposition import NMF

X = np.array([[5,3,0,1], 
              [4,0,0,1], 
              [1,1,0,5], 
              [1,0,0,4],
              [0,1,5,4]])

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

X1 = np.arange(1, 4+1)
Y1 = np.arange(1, 5+1)
X1, Y1 = np.meshgrid(X1, Y1)
Z1 = X
fig = plt.figure(1)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1, Y1, Z1, rstride=1, cstride=1, 
                       cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


# ------------------------------
model = NMF(n_components=2, alpha=0.001)

W = model.fit_transform(X)
H = model.components_
print ("W", W)
print ("H", H)


print("old",X)
print("rebuild")
rebuild = np.dot(W,H)
print( )


X1 = np.arange(1, 4+1)
Y1 = np.arange(1, 5+1)
X1, Y1 = np.meshgrid(X1, Y1)
Z1 = rebuild
fig = plt.figure(2)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1, Y1, Z1, rstride=1, cstride=1, 
                       cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

