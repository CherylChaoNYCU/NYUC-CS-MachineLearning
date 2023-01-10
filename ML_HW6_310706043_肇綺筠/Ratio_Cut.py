import cv2
import numpy as np
import os
from functions import*
from KMeans import*
from array2gif import write_gif
#print(cv2.__version__)
#parameter settings
imgp= 'image2.png'
flat_img,h,w = read_img(imgp)
gs = 0.001
gc = 0.001
k = 3
init_type = 'random_k'#'kmean++' #'random_k' 'gaussian'
#compute the kernel,W: the association matrix for each graph
W = kernel(flat_img,gs,gc) 
#degree matrix, sum up all degree and place them at the diagnol to form D
#axis = 1 as W = (datapoint,features)
D = np.diag(np.sum(W,axis=1))
#print(D.shape)
#Laplacian,unormalized -> ratiocut:Trace(H'LH)
L = D-W
#calculating this will take long time, remember to save the path and load it for next test
'''
eg_value,eg_vector = np.linalg.eig(L)
np.save('cluster{}_{}eg_value_gs{}_gc{}_RCut.npy'.format(k,imgp.split('.')[0],gs,gc),eg_value)
np.save('cluster{}_{}eg_vector_gs{}_gc{}_RCut.npy'.format(k,imgp.split('.')[0],gs,gc),eg_vector)
'''
#load the precomputed eigen values/vectors
eg_value = np.load('cluster{}_{}eg_value_gs{}_gc{}_RCut.npy'.format(2,imgp.split('.')[0],gs,gc))
eg_vector = np.load('cluster{}_{}eg_vector_gs{}_gc{}_RCut.npy'.format(2,imgp.split('.')[0],gs,gc))
print(eg_vector.shape)
sorted_u = np.argsort(eg_value) #sort the first k eigenvalue, this indicates the number of connected components(clusters)

U = eg_vector[:,sorted_u[1:1+k]]#pick the eigenvector of L from u2-uk to form U
#matches:the classes assigned to each datapoint(10000x1)
matches,segs = kmeans(U,k,h,w,type=init_type)
if k ==3:
    plotting(U[:,0],U[:,1],U[:,2],matches)

#write_gif(segs,'gaussian_clusters{}_{}_{}'.format(k,imgp.split('.')[0],'Ratio_Cut.gif'),fps=2)


cv2.waitKey(0)
cv2.destroyAllWindows()