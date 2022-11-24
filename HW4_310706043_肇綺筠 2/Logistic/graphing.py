import numpy as np
import matplotlib.pyplot as plt

def compose_confusion(A,b,ans):

    compare = np.hstack((b,ans)) #(pred , ground truth) -> (Nx1 , Nx1)
    tp = fp = fn = tn = 0
    #C0 as true C1 as false
    for  pairs in compare: #pair(0,0)  / (0,1) ... 
        if pairs[0] == pairs[1] == 0: #belongs to class 0
            tp+=1
        elif pairs[0] == pairs[1] == 1:
            tn+=1
        elif pairs[0] == 1 and  pairs[1] == 0:
            fp+=1
        else:
            fn+=1
    conf_matrix = np.empty((2,2))
    conf_matrix[0,0],conf_matrix[0,1],conf_matrix[1,0],conf_matrix[1,1] = tp,fn,fp,tn

    return conf_matrix

def print_confusion(m):
    print('Confusion Matrix:')
    print('            Predict cluster 1  Predict cluster 2')
    print('Is cluster 1         {:.0f}             {:.0f}    '.format(m[0,0],m[0,1]))
    print('Is cluster 2         {:.0f}             {:.0f}    '.format(m[1,0],m[1,1]))

    print('\n')
    print('Sensitivity (Successfully predict cluster 1):{}'.format(m[0,0]/(m[0,0]+m[0,1])))
    print('Specificity (Successfully predict cluster 2):{}'.format(m[1,1]/(m[1,0]+m[1,1])))

def ploting(c0,c1,title):
    plt.figure()
    plt.plot(c0[:,0],c0[:,1],'ro')
    plt.plot(c1[:,0],c1[:,1],'bo')
    plt.title(title)
    plt.show()