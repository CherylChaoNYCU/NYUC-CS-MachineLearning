from matplotlib import projections
import numpy as np
import cv2
from scipy.spatial.distance import pdist,squareform
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from array2gif import write_gif

#colors for ploting 
ColorMap = np.random.choice(range(256),size=(100,3)) #RGB values-> [0,0,255]...

'''
1. each pixel should be treated as data point ->  flatten the image to array
2. we have s(x) and c(x) two parameters
3. s(x) -> the size: Height, Width for image
4. c(x) -> color info (channel=3) C
'''
def read_img(p):
    img = cv2.imread(p)
    H,W,C = img.shape
    flatten = np.zeros((H*W,C))#10000x3
    for h in range(H):
        flatten[h*W:(h+1)*W] = img[h]
    return flatten,H,W

def kernel(x,gs=1,gc=1):
    n = len(x) #x:100x100x3,the length will be 10000
    s = np.zeros((n,2))#columns: width & height -> s records the h & w of each data point(hxw)
    #sqeuclidean -> vector間歐式距離的平方
    for i in range(n):
        s[i] = [i//100,i%100] #d1:(0,1),(0,2)...
    k = squareform(np.exp(-gs*pdist(s,'sqeuclidean')))*squareform(np.exp(-gc*pdist(x,'sqeuclidean')))
    return k

def visualize(classes,k,h,w): #x-> is classes!classes[i] = j(cluster) / k:# of clusters / height / width
    color = ColorMap[:k,:] #only k clusters, so we only need the first k color rows
    assign = np.zeros((h,w,3))#record the color assigning to specific data point
    for i in range(h):
        for j in range(w):
            assign[i,j,:] = color[classes[i*w+j]] #assign color to each datapoint, same cluster same color

    return assign.astype(np.uint8)#align with the pixel type

#plot three dimensional where cluster = 3
def plotting(x,y,z,classes):
    graph = plt.figure()
    #subplot1 with row=col=1
    ax = graph.add_subplot(111,projection='3d')
    for i in range(3):
        ax.scatter(x[classes==i],y[classes==i],z[classes==i])
    ax.set_xlabel('eigenvector d1')
    ax.set_ylabel('eigenvector d2')
    ax.set_zlabel('eigenvector d3')
    plt.show()


def gif_img(segs,path):
    #for i in range(len(segs)):
        write_gif(segs,path,fps=2)