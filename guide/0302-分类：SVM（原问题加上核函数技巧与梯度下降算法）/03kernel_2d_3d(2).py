# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from mpl_toolkits.mplot3d import Axes3D

# generating data
X, Y = make_circles(n_samples = 50, noise = 0.02)
#print("X",X)
#print("Y",Y)
# visualizing data
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c = Y, marker = '.')
plt.show()

# ------------------------------------------------------------
# adding a new dimension to X
t1 = X[:, 0].reshape((-1, 1))
t2 = X[:, 1].reshape((-1, 1))
t3 = 2**0.5 * t1*t2

X1 = t1*t1
X2 = t2*t2
X3 = t3
print("X1",X1)
print("X2",X2)
print("X3",X3)

t1 = np.hstack((X1, X2))
t1 = np.hstack((t1, X3))
X = t1
print("X", X)

# visualizing data in higher dimension
fig = plt.figure(2)
axes = fig.add_subplot(111, projection = '3d')
axes.scatter(X1, X2, X3, c = Y, depthshade = True)
plt.show()

# ------------------------------------------------------------

# create support vector classifier using a linear kernel
from sklearn import svm

svc = svm.SVC(kernel = 'linear')
svc.fit(X, Y)
w = svc.coef_
b = svc.intercept_

# plotting the separating hyperplane
z = lambda x,y: (-svc.intercept_[0]-svc.coef_[0][0]*x -svc.coef_[0][1]*y) / svc.coef_[0][2]

minx = X1.min()
maxx = X1.max()
miny = X2.min()
maxy = X2.max()
mymin = minx
mymax = maxx
if (maxy > maxx):
    mymax = maxy
if(miny< minx):
    mymin = miny
    
tmp = np.linspace(mymin,mymax,30)
x,y = np.meshgrid(tmp,tmp)

fig = plt.figure(3)
axes = fig.add_subplot(111, projection = '3d')
axes.scatter(X1, X2, X3, c = Y, depthshade = True)
axes.plot_surface(x, y, z(x,y),alpha = 0.05)
axes.view_init(28, 152)
plt.show()
