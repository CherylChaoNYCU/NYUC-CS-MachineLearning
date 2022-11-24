import numpy as np

from gradient import gradient_descent

#the convergence boundary
th = 1e-2
#we need to make sure if H inverse exist
def Newton_method(A,w,b,lr = 0.01):
    #recall the result of class note: phi.T D phi will be the hessian of log likelihood
    n = len(A)
    D = np.zeros((n,n))

    for i in range(n):
        D[i,i] = np.exp(-A[i]@w) / np.power(1+np.exp(-A[i]@w),2)
    H = A.T@D@A
    #make sure that H inverse exist...
    try:
        Hi = np.linalg.inv(H)
        #print('here')
    except np.linalg.LinAlgError as error:
        print(str(error))
        print('Hessian is singular, use steepest descent for instead')
        return gradient_descent(A,w,b)
    g = 1
    while np.sqrt(np.sum(g**2)) > th:
        g = A.T@(b - 1/(1+np.exp(-A@w)))
        w = w+ lr*(Hi@g)
        
    
    return w