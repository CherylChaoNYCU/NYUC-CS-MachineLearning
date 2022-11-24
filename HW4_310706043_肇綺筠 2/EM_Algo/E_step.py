import numpy as np

def find_w(A,Lambda,Prob):
    #to find w, we need Pi and 1-Pi

    Prob_complement = 1-Prob #Prob: classi's pixel probability showing"0" / 1-Prob: classi's prob showing"1"
    w = np.zeros((60000,10)) #each image has w0~w9 (predicted label)
    for i in range(60000):
        for j in range(10):
            w[i,j] = np.prod(A[i]*Prob[j]+(1-A[i])*Prob_complement[j])
        #each w should multiply its own lambda
    w = w*Lambda.reshape(1,-1)

    #sum of w0~w9*lambda (60000x1)
    sums = np.sum(w,axis=1).reshape(-1,1)
    sums[sums==0] = 1#avoid sum = 0, denominator should not be 0
    w = w/sums
    return w

