from tkinter import X
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from functions import *

#minimize ln p(y|theta), find the best parameter to define kernel that can best evaluate the relation between data points
def objective(x,y,beta):

    def obj(param): #find param that optimize kernel
    
        k = kernel(x,x,a=param[0],len_scale=param[1])+(1/beta)*np.identity(len(x))
        #l=np.linalg.cholesky(k)
        #return 0.5*y.reshape(1,-1)@np.linalg.inv(k)@y.reshape(-1,1)+np.sum(np.log(np.diag(l)))+0.5*len(x)*np.log(2*np.pi)#apply the  kernel into log likelihood function to find the best parameter
        return 0.5 * np.log(np.linalg.det(k)) + \
               0.5 * y.reshape(1,-1)@(np.linalg.inv(k)@y.reshape(-1,1)) + \
               0.5 * len(x) * np.log(2*np.pi)
    
    return obj 

x,y=load()
beta=5

obj_value=1e9
initials=[1e-3,1e-2,1e-1,0,1e1,1e2,1e3] #initial params for kernel
best_a = best_ls = 0
for ia in initials: #try and plug all initials into kernel, and then apply kernel to the likelihood function
    for ils in initials:
        ans = minimize(objective(x,y,beta),x0=[ia,ils],bounds=((1e-5,1e5),(1e-5,1e5))) #try other bounds
        if ans.fun < obj_value:
            obj_value=ans.fun
            best_a,best_ls = ans.x
            #print(best_a)
#print the best parameters for kernel
print('best alpha:',best_a)
print('best len_scale',best_ls)

#plug the best parameter we find into kernel ->k(x,x)

k = kernel(x,x,best_a,best_ls)+(1/beta)*np.identity(len(x))

linex = np.linspace(-60,60,num=600)
m_pred,v_pred = prediction(linex,x,y,k,beta,best_a,best_ls)
#we need to find standard deviation for graphing
m_pred = m_pred.reshape(-1)
v_pred = np.sqrt(np.abs(np.diag(v_pred)))
#ploting
#show all training data
plt.plot(x,y,'ro')
#testing data after prediction, we get mean = m_pred,var = v_pred
plt.plot(linex,m_pred,'k')
#95%confidence -> mean as the center += 2var
plt.fill_between(linex,m_pred+2*v_pred,m_pred-2*v_pred,color='powderblue')
plt.xlim(-60,60)
plt.show()