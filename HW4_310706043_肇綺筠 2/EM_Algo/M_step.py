import numpy as np

def update_L(w):
    #w:60000*10
    
    L = np.sum(w,axis=0)
    L/=60000
    return L.T

def update_P(A,w):
    #A[i] is xi of class note formula
    #recall the Pmle of M step
    #sum of wi, for denominator
    sums = np.sum(w,axis=0)
    sums[sums==0]=1
    #nominator: sum of xi*wi
    w_norm = w/sums
    P = A.T@w_norm
    return P.T
   

