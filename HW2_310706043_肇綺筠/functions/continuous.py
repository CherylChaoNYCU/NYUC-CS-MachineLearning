import math
import numpy as np

def Variance(arr,e):
    v = np.var(arr)
    return v if v!=0 else e



def Gaussian(x,mu,var):
    ans = ((1/math.sqrt(2*math.pi*var))*math.exp((-(x-mu)**2)/(2*var)))
    return ans


def cont_prob(trainX,trainY,e):
    con = np.zeros((10,28*28,256)) #256 is the size of pixel value
    
    for class_ in range(10):
        arr = trainX[trainY == class_] #A storing pixel values (0-256), 10 classes (0-9)
        #arr shape:(number of class c pics x 28*28)
        for i in range(28*28):
            mu = np.mean(arr[:,i]) #Mu for gaussian
            var = Variance(arr[:,i],e)
            for j in range(256):
                con[class_,i,j] = Gaussian(j,mu,var)
    return con

def cont_test(test_pic,pix_prob,prior,testx,testy,ep):
    err = 0
    for i in range(test_pic):
        p = np.zeros(10)
        for class_ in range(10):

            for dim in range(28*28):
                p[class_]+=np.log(max(ep,pix_prob[class_,dim,int(testx[i,dim])]))
           
            p[class_]+=np.log(prior[class_])
        
        p/=np.sum(p) #normalizing
        print('Posterior (in log scale):')
        for class_ in range(10):
            print('{}: {}'.format(class_,p[class_])) #show the likelihood of data x belonging to class_
        pred = np.argmin(p)
        print('predictions:{}, answers:{}'.format(pred,testy[i]))
            #print('--------------------')
        print()

        if pred!=testy[i]:
            err+=1
    print('Error Rate:{}'.format(err/test_pic))
    print()
        


