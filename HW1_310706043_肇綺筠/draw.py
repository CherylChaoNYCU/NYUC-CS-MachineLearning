import matplotlib.pyplot as plt
import numpy as np

def model_line(par):
    par = par.reshape(-1)
    n = len(par)-1
    print('Fitting line: ',end='')

    for i in range(n,0,-1): #descending order
        print(par[i],'x^',i,'+',end='')
    
    print(par[0]) #constant

def draw_(xx,b,par_rlse,par_nt):

    #draw rlse
    plt.subplot(2,1,1) #seperate to two graph
    plt.title('rLSE')
    plt.plot(xx,b,'bo') #'ro' is red dots
    x_min=min(xx)
    x_max=max(xx)
    x=np.linspace(x_min-1,x_max+1,500)
    y=np.zeros(x.shape) #dummy for answers after pluging into tuned function
    for i in range(len(par_rlse)):
        y+=par_rlse[i]*np.power(x,i)
    plt.plot(x,y,'-k') #'k' is the black line

    print('\n')

    #newton
    plt.subplot(2,1,2)
    plt.title('Newton\'s Method')
    plt.plot(xx, b, 'ro') #plotting real data
    y = np.zeros(x.shape)
    for i in range(len(par_nt)):
        y += par_nt[i] * np.power(x, i)
    plt.plot(x, y, '-k') #plotting data after pluging into the tuned function
    plt.show()