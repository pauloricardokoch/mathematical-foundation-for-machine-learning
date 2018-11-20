from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np


# create a figure and a subplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# generate 2d meshgrid
nx, ny = (50, 50)

x = np.linspace(-10, 10, nx)
y = np.linspace(-10, 10, ny)


xv, yv = np.meshgrid(x, y)

# define a function to plot
def f(x, y):
	return x ** 2 + 2 * y


# calculate de z value for each x,y point
z = f(xv, yv)

print(xv)
print(xv.shape)

print(yv)
print(yv.shape)

print(z)
print(z.shape)

ax.plot_wireframe(xv, yv, z, rstride=1, cstride=1)
plt.show()