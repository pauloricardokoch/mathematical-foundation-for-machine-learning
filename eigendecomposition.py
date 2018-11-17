import sys
import numpy as np

from numpy import linalg

# define an array
a = np.arange(9) - 3
print a

b = a.reshape((3, 3))
print b

# euclidian L2 norm - default
print np.linalg.norm(a)
print np.linalg.norm(b)

# the frogenius norm is the L2 norm for a matrix
print np.linalg.norm(b, 'fro')

# the max norm (P = infitity)
print np.linalg.norm(a, np.inf)
print np.linalg.norm(b, np.inf)

# vector normalization - normalization to produce a unit vector
norm = np.linalg.norm(a)
a_unit = a / norm
print a_unit

# the magnitude of a unti vector is equal to 1
mag = np.linalg.norm(a_unit)
print mag

# find the eigenvalues and eigenvector for a simple square matrix
a = np.diag(np.arange(1, 4))
print a

eigenvalues, eigenvectors = np.linalg.eig(a)

print eigenvalues
print eigenvectors

# the eigenvalue w[i] corresponds to the eigenvactor v[:,i]
print 'Eigenvalue: {}'.format(eigenvalues[1])
print 'Eigenvector: {}'.format(eigenvectors[:, 1])

# verify eigendecomposition
matrix = np.matmul(np.diag(eigenvalues), np.linalg.inv(eigenvectors))
output = np.matmul(eigenvectors, matrix).astype(np.int)
print output

# import necessary matplot libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline

# plot the eigenvectors
origin = [0, 0, 0]
fig = plt.figure(figsize=(18,10))
fig.suptitle('Effects of Eigenvalues and Engenvectors')
ax1 = fig.add_subplot(121, projection = '3d')

ax1.quiver(origin, origin, origin, eigenvectors[0, :], eigenvectors[1, :], eigenvectors[2, :], color = 'k')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
ax1.set_zlim([-3, 3])
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')
ax1.view_init(15, 30)
ax1.set_title('Before Multiplication')

# multiply original matrix by eigenvectors
new_eig = np.matmul(a, eigenvectors)
ax2 = plt.subplot(122, projection = '3d')
ax2.quiver(origin, origin, origin, new_eig[0, :], new_eig[1, :], new_eig[2, :], color = 'k')

# add the egeinvalues to the plot
ax2.plot((eigenvalues[0] * eigenvectors[0]), (eigenvalues[1] * eigenvectors[1]), (eigenvalues[2] * eigenvectors[2]), 'rX')
ax2.set_xlim([-3, 3])
ax2.set_ylim([-3, 3])
ax2.set_zlim([-3, 3])
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.set_zlabel('Z axis')
ax2.view_init(15, 30)
ax2.set_title('After Multiplication')

# show the plot
plt.show()
