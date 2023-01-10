import numpy as np

def load(path='input.data'):
    x=[]
    y=[]
    f = open(path,'r')
    for l in f.readlines():
        pts = l.split(' ')
        x.append(float(pts[0]))
        y.append(float(pts[1]))
    f.close()
    #transfer x,y to numpy array
    x = np.asarray(x)
    y = np.asarray(y)

    return x,y


#rational quadratic kernel function for GP
def kernel(x1,x2,a=1,len_scale=1):
    sq_error = np.power(x1.reshape(-1,1)-x2.reshape(1,-1),2.0)
    kernel = np.power((1+sq_error/2*a*len_scale**2),-a)

    return kernel

#x_test is the testing data x ranging from -60~60 that we want to predict, and x is the training data given by HW
#k-> k(x,x) relation of training data

def prediction(x_test,x,y,k,beta,a=1,len_scale=1):
    #kxx_test -> k(x,x*) relation between test and training
    kxx_test = kernel(x,x_test,a=1,len_scale=1)
    #k(x*,x*)
    kx_testx_test = kernel(x_test,x_test,a=1,len_scale=1)
    #mean(x*)=k(x,x*).T*inverse(k(x,x))*y
    mean = kxx_test.T @ np.linalg.inv(k)@y.reshape(-1,1)
    #k(x*,x*)+1/beta should become matrix
    vars = kx_testx_test+(1/beta)*np.identity(len(kx_testx_test))-kxx_test.T@np.linalg.inv(k)@kxx_test

    return mean,vars