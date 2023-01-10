import numpy as np
from functions import *

eps = 1e-8 #the EM algo convergence thresold


#Gram,k,HEIGHT,WIDTH,initType=k_means_initType
#reminder:input x is already projected to the vector space by kernel function
def ini_mean(x,k,type): #x-> k(x,x)the value of kernel :(datapoints,features)/ clusters:3 / type
    
    #print(x.shape)
    cluster = np.zeros((k,x.shape[1])) #record the cluster mean,total:three clusters x data features
    
    if type == "kmean++":
        #the first cluster mean, randomly choose a datapoint as mean(center of the cluster)
        cluster[0] = x[np.random.randint(low=0,high=x.shape[0],size=1),:] 

        for c in range(1,k): #len(x) = number of datapoints
            d = np.zeros((len(x),c)) #d records the distance between data point and the center of each cluster
            for i in range(len(x)): #pick one data
                for j in range(c): #calculate the distance between the data and  centers (mean) of all other clusters
                    d[i,j] = np.sqrt(np.sum((x[i]-cluster[j])**2))
            d_min = np.min(d,axis=1) #filter out the min distance 
            sum = np.sum(d_min)*np.random.rand() 
            for i in range(len(x)):
                sum-=d_min[i] 
                if sum<=0: #the min distance makes sum <=0(max), we should select it as centorid
                    cluster[c] = x[i]#find the new mean and update the center of cluster
                    break
    elif type == 'random_k': #randomly choose k numbers as center
        rand = np.random.randint(low=0,high=x.shape[0],size=k)
        cluster = x[rand,:]
    else: #gaussian sample, use the mean and  variance of dataset to find the center of the cluster
        mean = np.mean(x,axis=0)#we need to calculate the mean according to 0 axis(feature)of data
        std = np.std(x,axis=0)
        for c in range(x.shape[1]):#c traverse the feature of x!
            cluster[:,c] = np.random.normal(mean[c],std[c],size=k)#cluster c use mean[c]/std[c]

    return cluster


def kmeans(x,k,h,w,type='random',gifpath='default.gif'):

    mean = ini_mean(x,k,type) #return the initial mean of each cluster
    #mean:(# of clusters , the data(mean)feature)
    #record the class of each datapoint(100x100)
    Classes = np.zeros(len(x),dtype=np.uint8)
    segs = []
    diff = 1e8
    count=1
    while diff > eps: 
        #the E-step, find alphank
        for i in range(len(x)):
            dist = []
            for j in range(k):
                dist.append(np.sqrt(np.sum((x[i]-mean[j])**2))) #there will only be k distances in dist
            Classes[i] = np.argmin(dist) #classes[i] records the index of min distance between data and class

        #the M-step:update the center(mean)of the cluster after first classification
        new_mean = np.zeros(mean.shape)
        for i in range(k): #traverse all clusters
            match = np.argwhere(Classes == i).reshape(-1) #this will return the data index that belongs to cluster i
            for j in match:#traverse all the datapoint in cluster i and find the new mean
                new_mean[i]=new_mean[i]+x[j]
            if len(match)>0:
                new_mean[i]=new_mean[i]/len(match)
        diff = np.sum((new_mean-mean)**2)
        mean = new_mean 

        seg = visualize(Classes,k,h,w) #do color assignment to each datapoint within different clusters
        segs.append(seg)
        print('step:{}'.format(count))

        for i in range(k):#record how many data is classified in specific cluster during different steps(iteration)
            print('cluster k = {}: data {}'.format(i+1,np.count_nonzero(Classes == i)))
        print('parameter diff = {}'.format(diff))
        print('======================')
        cv2.imshow('',seg)#the color
        cv2.waitKey(1)
        count+=1
    
    return Classes,segs











            


            
            



