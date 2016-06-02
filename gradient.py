
import numpy as np
import random
steps = 10000
precision = 1.068
Y = np.array([[5, 1, 4, 5, 1], [0, 5, 2, 1, 4], [1, 4, 1, 1, 2],\
              [4, 1, 5, 5, 4], [5, 3, 4, 5, 4], [1, 5, 1, 1, 1], \
              [5, 1, 0, 5, 4]])

R = np.array([[1,1,1,1,1],[0,1,1,1,1],[1,1,1,1,1],
              [1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],
              [1,1,0,1,1]])

random.seed(666)
U = np.random.random((7,2))


V = np.random.random((5,2))
V_t = np.transpose(V)

for step in xrange(steps):
    V_t = np.transpose(V)
    U_new = U + 1/33.0 * .10 * (np.dot((R*(Y-(np.dot(U, V_t)))), V))
    V_new = V + 1/33.0 * .10 * (np.dot(np.transpose(R*(Y-(np.dot(U, V_t)))), U))
    U = U_new
    V = V_new
    h = np.dot(U_new, np.transpose(V_new))

    error = 0.0
    error = (1/33.0)*(np.sum((Y - h)**2))

    if abs(error) < precision:
        print "minima_reached"
        break

print U, V
