import numpy as np
import matplotlib.pyplot as plt
from functions import *
from gradient import gradient_descent
from graphing import *
from newton import Newton_method


N = int(input('N: '))
mx1,my1 = [int(x) for x in input('mx1 & my1: ').split()]
mx2,my2 = [int(x) for x in input('mx2 & my2: ').split()]
vx1,vy1 = [int(x) for x in input('vx1 & vy1: ').split()]
vx2,vy2 = [int(x) for x in input('vx2 & vy2: ').split()]

#generating two class data(D1,D2 gaussian)

D1 = sample(mx1,my1,vx1,vy1,N)
D2 = sample(mx2,my2,vx2,vy2,N)
ploting(D1,D2,'Ground Truth')
#the design matrix for data points (phi in the class note)
A = DM(D1,D2)
#b: ground trurh. the class of D1(->C0) and D2(->C1)
b = vector_b(N)

#initialze the parameters
w = np.random.rand(3,1)
#print('before:',w)
#train model, find the best parameter of LR model
w = gradient_descent(A,w,b,lr=0.001)
#print(w)


print('Gradient descent:\n')
print('w:\n')
print(w[0])
print(w[1])
print(w[2])

#do prediction with the model with updated w
ans = predict(A,w)
conf_matrix = compose_confusion(A,b,ans)
c0,c1 = classifying(A,ans) #classify all data based on our prediction
print_confusion(conf_matrix)
ploting(c0,c1,'Gradient descent')

#newton update

w = np.random.rand(3,1)
#print('before:',w)
w  = Newton_method(A,w,b,lr = 0.001)

print('============================')
print('Newton\s Method:\n')
print('w:\n')
print(w[0])
print(w[1])
print(w[2])

ans = predict(A,w)
conf_matrix = compose_confusion(A,b,ans)
c0,c1 = classifying(A,ans) #classify all data based on our prediction
print_confusion(conf_matrix)
ploting(c0,c1,'Newton\s Method:\n')