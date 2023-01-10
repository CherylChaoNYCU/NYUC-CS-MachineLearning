import csv
import sys
import time
import numpy as np
from libsvm.svmutil import *


kernel = {
    'linear':0,
    'polynomial':1,
    'RBF':2,
}

def load(fn):
    with open(fn,'r') as f:
        data = list(csv.reader(f))
        data = np.array(data)

    return data


def compare_ACC(y,x,search,opt_acc,opt_search):
    print('current input search:{}'.format(search))
    acc = svm_train(y,x,search)
    if acc > opt_acc:
        return acc,search
    else:
        return opt_acc,opt_search

# def train(y,x,kernel):
#     #print(kernel)
#     return svm_train(y,x,f'-t {kernel} -q')


def grid_search(x,y):
    optimal_search='' #initialized, dummy...
    optimal_acc=0
    #cost 'c' for C-SVC: describes how hard the data "within" the margins are penalized
    #the larger the 'c' is, the harder the data will be penalized
    costs = [0.001,0.01,0.1,1,10,10]
    gammas=[0.001,0.01,0.1,1] #the parameter for RBF kernel (var)
    #count=0
    for k in kernel: #traverse different kernels
        for c in costs:
            if k =='linear': #the linear kernel: u’*v
                search = f'-t {kernel[k]} -c {c} -v 3 -q' #-t kernel type,-c cost, -v crossvalid k=4
                #count+=1
                optimal_acc,optimal_search = compare_ACC(y,x,search,optimal_acc,optimal_search)
            elif k == 'polynomial': #the poly kernel:(gamma*u’*v + coef0)^degree -> we need gamma,coef0 and degree parameters
                for g in gammas:
                    for d in range(2,5):
                        for coef in range(0,4):
                            search = f'-t {kernel[k]} -g {g} -c {c}  -d {d} -r {coef} -v 3 -q' #-t kernel type,-c cost, -v crossvalid k=4
                            optimal_acc,optimal_search = compare_ACC(y,x,search,optimal_acc,optimal_search)

            elif k == 'RBF': # the RBF kernel: exp(-gamma*|u-v|^2), we need gamma parameter
                for g in gammas:
                     search = f'-t {kernel[k]} -g {g} -c {c}  -v 3 -q' #-t kernel type,-c cost, -v crossvalid k=4
                     optimal_acc,optimal_search = compare_ACC(y,x,search,optimal_acc,optimal_search)
    print('Best Accuracy with cross-validation k=4:{}'.format(optimal_acc))
    print('Best Combination:{}'.format(optimal_search))

    return optimal_search


def linear_kernel(x1,x2): #finding:k(x,x),k(x,x*)...
    K = x1@x2.T
    return K

def RBF_kernel(x1,x2,g):
    distx1x2 = np.sum(x1**2,axis=1).reshape(-1,1)+np.sum(x2**2,axis=1)-2*x1@x2.T #expand:|x1-x2|^2
    k = np.exp((-g*distx1x2)) # exp(-gamma*|u-v|^2)
    return k



                


if __name__ == '__main__':

    # print('1 for part1: compare 3 kernel performance')
    # print('2 for part2: C-SVC grid search')
    # print('3 for part3: linear kernel + RBF kernel')

    xtrain = load('X_train.csv').astype(np.float64)
    ytrain = list(load('Y_train.csv').astype(np.int32).ravel()) #flatten the label as list
    xtest = load('X_test.csv').astype(np.float64)
    ytest = list(load('Y_test.csv').astype(np.int32).ravel())

    if sys.argv[1] == '1': #comparing three kernels
        for k in kernel:
            print(f'kernel type: {k}')
            #svm_train returns three things:model | ACC | MSE
            model = svm_train(ytrain,xtrain,f'-t {kernel[k]} -q')#transfering the kernel type into svm train function
            #svm_predict returns:(predicted labels, acc+mse, alist of decision value)
            result = svm_predict(ytest,xtest,model)
    
    elif sys.argv[1] == '2': #use C-SVC with grid search for best model
        search=grid_search(xtrain,ytrain)
        print('The best combination ACC of:{} after testing:'.format(search))
        # model = svm_train(ytrain,xtrain,search)
        # result = svm_predict(ytest,xtest,model)#show accuracy of  the best search
   
    elif sys.argv[1] == '3': #do linear + RBF
        linear_kxx = linear_kernel(xtrain,xtrain)#k(x,x)for linear
        #default gamma ->1/features = 1/28*28 = 1/784pixels
        rbf_kxx = RBF_kernel(xtrain,xtrain,1/784)
        #while we are training, we need to throw the data that have already passed into feature space(kernel)into svm model! since we customized the kernel function
        linear_kxxs = linear_kernel(xtrain,xtest).T #for testing! k(x,x*)
        rbf_kxxs = RBF_kernel(xtrain,xtest,1/784).T

        #now stacking 2 kernels
        k_stack = np.hstack((np.array(1,5001).reshape((-1,1)),linear_kxx+rbf_kxx)) #combining 2 kernels for training,store them for each training data into matrix
        k_stack_s = np.hstack((np.array(1,2501).reshape((-1,1)),linear_kxxs+rbf_kxxs)) #combining 2 kernels for testing

        search = '-t 4 -q' # 4 is for "precomputed kernel"
        model = svm_train(ytrain,k_stack,search) #kernel combination for training
        result = svm_predict(ytest,k_stack_s,model) #kernel combination for testing







