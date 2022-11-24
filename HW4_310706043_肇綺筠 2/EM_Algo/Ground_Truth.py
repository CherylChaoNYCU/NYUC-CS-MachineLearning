import numpy as np
import math


def discrete_prob(trainX,trainY): 
    
    label_count = np.zeros(10)
    for i in trainY:
        #treat classes 0-9 as index
        label_count[i]+=1

    #real answer distribution
    ground_truth = np.zeros((10,784))

    for i in range(60000):
        class_ = trainY[i]
        for j in range(784):
            if trainX[i,j] == 1:#it will only be 0 or 1 as the spec asked for 2 bins
                ground_truth[class_,j]+=1
    
    ground_truth = ground_truth/label_count.reshape(-1,1)

    return ground_truth
    










