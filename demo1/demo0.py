'''
editor: Jones
date: 10/25,2018
content: demo0
'''

# print 'Hello World!!!

import numpy as np 

# a =np.array([10,20,30,40])
# b = np.arange(4)
# c = 10*np.tan(a)

# a = np.array([[1,1], [0,1]])
# b = np.arange(4).reshape((2,2))

# c = a*b
# c_dot = np.dot(a, a)
# c_dot_2 = b.dot(a)

# a = np.random.random((2,4))

# print a
# print np.sum(a, axis=1)
# print np.min(a, axis=0)
# print np.max(a, axis=1)


# print c_dot
# print c_dot_2


A = np.arange(2,14).reshape((3,4))
print A[0]

# print np.argmin(A)
# print np.argmax(A)
print np.cumsum(A)[-1]
print A.mean()
