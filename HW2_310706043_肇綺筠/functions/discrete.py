import numpy as np
import math

#get the pixel value prob
#return number of class/num of dimension 28*28/num of pixel value(32 bin)
def discrete_prob(trainX,trainY): #transfer images,labels
    
    de = np.zeros((10,28*28,32)) #class 0-9 / dim:28*28 / pixel to 32 bins, storing the probability of each pixel(data points)
    
    for i in range(len(trainX)): #traverse each image (60000)
        class_ = trainY[i] #image:label
        for dim in range(28*28):#image dimensions
            de[class_][dim][int(trainX[i,dim])//8]+=1 #transfering to bins, 0-7/8-15...group in one bin
    for class_ in range(10):
            for dim in range(28*28):
                c = 0
                for b in range(32):
                    c+=de[class_][dim][b] #sum up bin-form pixel values
                de[class_][dim][:]/=c #return the pixel value probability
        
    return de

def discrete_test(test_pic,pix_prob,prior,testx,testy): #test_pic = number of testing pictures
    err = 0
    for i in range(test_pic):
        p = np.zeros(10)
        for class_ in range(10):
            for dim in range(28*28):
                #apply max likelihood to do the bayesian classifier(naive, each data is independent)
                p[class_] += np.log(max(1e-4,pix_prob[class_,dim,int(testx[i,dim])//8])) #p(x|c1), a term at the posterior nominator
            p[class_]+=np.log(prior[class_]) #doing log(p(x|c1)*p(c1))

        p/=np.sum(p) #divide the marginal prob of all data

        print('Postirior (in log scale):')
        for class_ in range(10):
            print('{}: {}'.format(class_,p[class_])) #show the likelihood of data x belonging to class_
                #reminders: probability is negative, so choose the min one
        pred = np.argmin(p)
        print('predictions:{}, answers:{}'.format(pred,testy[i]))
        print()

        if pred!=testy[i]:
            err+=1
    print('Error Rate:{}'.format(err/test_pic))
    print()








