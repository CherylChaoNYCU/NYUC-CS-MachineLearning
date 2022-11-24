import numpy as np
import random
import math


#from HW3.1
def Gaussian_Generator(m,s):
    #U,V are independent rv
    U = random.random()
    V = random.random()
    Z = math.sqrt(-2*math.log(U))*math.cos(2*math.pi*V)
    data = m+s**(0.5)*Z 
    return data

def sample(mx,my,vx,vy,N):
    tmp = np.empty((N,2))

    for i in range(N):
        #x,y are independently sample from gaussian
        tmp[i,0] = Gaussian_Generator(mx,vx)
        tmp[i,1] = Gaussian_Generator(my,vy)
    #print(tmp)
    #print('###')
    return tmp

#the design matrix, stacking D1 & D2
def DM(d1,d2):
    A = np.ones((2*len(d1),3)) #3: constant/x^1/x^2
    A[:,1:] = np.vstack((d1,d2))

    #print(A)

    return A

#b is ground truth: D1 belongs to class 0, D2 belongs to class1
#so the first 50 points is "0", and the other half(50~100) is "1"
def vector_b(N):
    b = np.zeros((2*N,1))
    b[N:] = np.ones((N,1))

    #print(b)
    return b

def predict(A,w):
    #throw points into the model
    #if the output >0 => class 0, otherwise => class1
    N = len(A)
    pred = np.empty((N,1))
    for i in range(N):
        pred[i] = 0 if A[i]@w <0 else 1
    
    return pred

def classifying(A,ans): #answer is the prediction outcome after we throw data points into our model(Aw)
    c0 = []
    c1 = []
    total_pt = len(A)
    for i in range(total_pt): #classifying data in design matrix
        if ans[i] == 0:
            c0.append(A[i,1:])
        else:
            c1.append(A[i,1:])
    # print('c0')
    # print(c0)

    # print('c1')
    # print(c1)
    return (np.array(c0),np.array(c1))


        

