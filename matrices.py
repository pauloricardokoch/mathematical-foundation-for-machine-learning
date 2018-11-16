import sys
import numpy as np

print 'Python: {}'.format(sys.version)
print 'NumPy: {}'.format(np.__version__)

#defining a scalar
print '\ndefining a scalar'
x = 6
print x
print '\n'

#defining a vector
print '\ndefining a vector'
x = np.array((1, 2, 3))
print x

print 'Vector dimensions: {}'.format(x.shape)
print 'Vector size: {}'.format(x.size)

#defining a matrix
print '\ndefining a matrix'
x = np.matrix([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])
print x

print 'Matrix dimensions: {}'.format(x.shape)
print 'Matrix size: {}'.format(x.size)


x = np.ones((3, 3, 3))
print '\ndefining a tensor with ones'
print x

print 'Tensor dimensions: {}'.format(x.shape)
print 'Tensor size: {}'.format(x.size)

#indexing
x = np.ones((5, 5), dtype = np.int)
print '\n'
print x

x[0, 1] = 2
print '\n'
print x

x[:, 0] = 3
print '\n'
print x

x[:,:] = 5
print '\n'
print x


#matrix operations
a = np.matrix([[1, 2],[3, 4]])
b = np.ones((2,2), dtype = np.int)


print '\n'
print a

print '\n'
print b

#element wise sum
c = a + b
print '\n'
print c


#element wise subtraction
c = a - b
print '\n'
print c

#matrix multiplication
c = a * b
print '\n'
print c

#matrix transpose
a = np.array(range(9))
a = a.reshape(3, 3)
print '\n'
print a

a = a.T
print '\n'
print a

#tensor
a = np.ones((3,3,3,3,3,3,3,3,3,3))
print '\n'
print a.shape
print len(a.shape)
print a.size
