import cv2
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
init_type = 'kmean++'#'kmean++' #'random_k' 'gaussian'
#gif_p = os.path.join('GIF','{}_{}'.format(imgp.split('.')[0],'kernel_k_means.gif'))
#compute the kernel
K = kernel(flat_img,gs,gc)
matches,segs = kmeans(K,k,h,w,type=init_type)

#segs is the clustering result during each iteration
write_gif(segs,'gaussian_clusters{}_{}_{}'.format(k,imgp.split('.')[0],'kernel_k_means.gif'),fps=2)


cv2.waitKey(0)
cv2.destroyAllWindows()