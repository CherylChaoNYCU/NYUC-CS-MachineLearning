import numpy as np
import matplotlib.pyplot as plt
import os
from draw import model_line
from draw import draw_

#find  x = ATA inverse ATb through rLSE and newton's method
#A is the matrix of input data, b is the parameter of model, x is our model parameters

#這裡使用LU分解求A的逆矩陣, as computing inverse of L U saves a lot of time
#A = LU
#A inverse = U （下三角矩陣）inverse * L（上三角矩陣）inverse
#ref: https://blog.csdn.net/weixin_43425490/article/details/120269793?spm=1001.2101.3001.6650.13&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-13-120269793-blog-39553809.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-13-120269793-blog-39553809.pc_relevant_default&utm_relevant_index=17
#ref2 PALU:https://blog.csdn.net/weixin_28950415/article/details/113067512?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-113067512-blog-120269793.t5_download_50w&spm=1001.2101.3001.4242.2&utm_relevant_index=4
#ref3: LU decomposing: https://www.youtube.com/watch?v=BFYFkn-eOQk


def inverse(A):
    m,n = A.shape

    P,L,D,U = LUP_decomp(A) #using PA = LU, where P is the transpose matrix

    #inverse L
    Li = L_Inverse(L)
    #inverse U
    Ui = U_Inverse(U)

 
    Di = D.copy() # 把U變回來
    for i in range(m):
        Di[i,i] = 1/Di[i,i]
    
    return Ui@Di@Li@P



def LUP_decomp(A):
    m,n = A.shape
    U = A.copy()
    D = np.identity(m) #上三角矩陣的主元素
    L = np.identity(m) #we want the diognal of L to be "1"
    P = np.identity(m)
     #PA = LU
    for i in range(m):
        
        maxA=abs(U[i,i]) #max element of col i
        maxRow=i 
        for k in range(i+1,m):
            if abs(U[k,i])>maxA:
                maxA=U[k,i]
                maxRow=k
        #swap rowi and the row with max element
        U[[i,maxRow],i:]=U[[maxRow,i],i:] 
        P[[i,maxRow], :] = P[[maxRow, i], :] #置換一些U裡面的元素，在做高斯消去
        #高斯eliminate找出上下三角形矩陣
        for k in range(i+1,m):
            c=-U[k,i]/U[i,i]
            U[k,i:]=U[k,i:]+c*U[i,i:]
            L[k:,i]=L[k:,i]-c*L[k:,k]

    # from U to D*U -> U的主元素全都會變成1,算inverse就會很快
    #  1 all over the main diagonal,no need to apply the row operations to get the inverse, you only need to change the signs of the off-diagonal elements.
    for i in range(m): 
        D[i,i]=U[i,i]
        U[i,i:]=U[i,i:]/D[i,i]

    return P,L,D,U



#ref: LU inverse: http://home.cc.umanitoba.ca/~farhadi/Math2120/Inverse%20Using%20LU%20decomposition.pdf


def L_Inverse(A):

    m,n = A.shape
    Ai = np.identity(m) #mxm identity matrix
    for i in range(m-1):
        for k in range(i+1,m):
            Ai[k,:]-=Ai[i,:]*A[k,i]  #A[1,:]access all elements in the first row
    
    return Ai





def U_Inverse(A):
    m,n=A.shape
    Ai=np.identity(m)
    for i in range(m-1,0,-1): #from right to left
        for k in range(i-1,-1,-1):
            Ai[k,:]-=Ai[i,:]*A[k,i]
    return Ai



def rlse(A,Lambda,b):#A: dataset b: ground truth
    m,n = A.shape
    x = inverse(A.T@A+Lambda*np.identity(n))@A.T@b #the best solution of parameter after calculating the gradient of rLSE
    #.T is transpose in numpy, @ is matrix mul in numpy
    loss_value = loss(A,x,b) #x is the parameter of function we found after gradient. Now we plug our data A into this basis-poly with parameter x , and 
                             #compute the loss with ground truth b (point y of input data)
    return x,loss_value

def loss(A,x,b): #LSE
    return np.sum(np.square(A@x-b))




def newton(A,b):
    m,n = A.shape

    x_in = np.random.rand(n,1) #initializing basis poly parameters
    t = 1000

    while t > 1e-5:
        x1 = x_in - inverse(2*A.T@A)@(2*A.T@A@x_in-2*A.T@b)
        t = abs(np.sum(np.square(x1-x_in))/n) #the deviation of two parameters (new xn+1 and previous xn), when their distance is smaller than 1e-6, it implies that the update doesn't make big change anymore => stop updating
        x_in = x1

    loss_val = loss(A,x_in,b)
    return x_in,loss_val

def loss(A,x,b):
     return np.sum(np.square(A@x-b))









xx = []
y = []

path = input('file path: ')
name = input('file name: ')

filepath = os.path.join(path,name)
fp=open(filepath,'r')
line=fp.readline()

#read the points from testfile.txt

while line:
    a,b=line.split(',') #x,y split by ','
    xx.append(float(a)) #data point x
    y.append(float(b)) #data point y
    line=fp.readline()

#convert input data into array

xx = np.asarray(xx,dtype = 'float').reshape((-1,1)) #reshape to only one column
b = np.asarray(y,dtype = 'float').reshape((-1,1)) #b is the real answer (numbers of y)

#now we do the rLSE,newton's method using while loop

while True:
    basis_poly = int(input('n: '))
    Lambda = int(input('lambda: ')) #for rLSE, the penalty of function
    print('poly_basis size: ',basis_poly)
    
    #matrix A with row = size of data point, col = basis poly (number of parameters)
    #as we need to do: Ax - b
    A = np.empty((len(xx),basis_poly))
    for j in range(basis_poly):
        A[:,j] = np.power(xx,j).reshape(-1) #record the answer at the jth in each row
        #recording the answers after pluging points into basis function

    #implement rLSE

    par_rlse,loss_rlse = rlse(A,Lambda,b)
    print('LSE result:')
    model_line(par_rlse)
    print('Total error: ',loss_rlse)
    print()



    #implement newton's method

    par_nt,loss_nt = newton(A,b)
    print('Newton\'s Method result:')
    model_line(par_nt)
    print('Total error: ',loss_nt)
    print()

    #compare parameters from rlse + newton
    draw_(xx.reshape(-1), b.reshape(-1), par_rlse.reshape(-1),par_nt.reshape(-1))