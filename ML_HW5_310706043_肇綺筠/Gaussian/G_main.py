import numpy as np
import matplotlib.pyplot as plt
from functions import *


x,y = load()
beta=5



#recall: the kernel for predicting y* -> K* = k(x*,x*)+inverse(beta)
# L is the kernel length scale and a determines the weighting of large and small scale variations.
K = kernel(x,x,a=1,len_scale=1)+1/beta*np.identity(len(x))

linex = np.linspace(-60,60,num=600)
m_pred,v_pred = prediction(linex,x,y,K,beta,a=1,len_scale=1)
#we need to find standard deviation for graphing
m_pred = m_pred.reshape(-1)
v_pred = np.sqrt(np.diag(v_pred))
#ploting
#show all training data
plt.plot(x,y,'ro')
#testing data after prediction, we get mean = m_pred,var = v_pred
plt.plot(linex,m_pred,'k')
#95%confidence -> mean as the center += 2var
plt.fill_between(linex,m_pred+2*v_pred,m_pred-2*v_pred,color='powderblue')
plt.xlim(-60,60)
plt.show()