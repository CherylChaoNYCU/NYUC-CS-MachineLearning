import numpy as np

#the convergence boundary
th = 1e-2
#g -> partial derivative of log likelihood( if root of square sum is less than the threshold,
# the improvement is insignificant
# stop updating)
#careful: overflow problem...
def gradient_descent(A,w,b,lr = 0.01):
    g = 1
    while np.sqrt(np.sum(g**2)) > th:
        g = A.T@(b - 1/(1+np.exp(-A@w)))
        w = w+lr*g
    
    return w