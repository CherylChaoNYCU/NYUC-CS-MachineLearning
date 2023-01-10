import matplotlib.pyplot as plt
import numpy as np
import os 

#(picked_data, picked_filename, 'pca_eigenface', W, mu)
def plottting(picked_d,picked_fn,title,W,mean=None):
    if mean is None: 
        mean = np.zeros(picked_d.shape[1])
    #PCA 
    Z = (picked_d - mean) @ W
    reconstruct = Z@W.T+mean

    #make directory to save the eigenface graph
    folder = f"{title}"
    os.mkdir(folder)
    os.mkdir(f'{folder}/{title}')
 
    #show eigenface, combining 25 eigenfaces into one graph
    if W.shape[1] == 25:
        plt.clf
        for i in range(5):
            for j in range(5):
                idx = i*5+j
                plt.subplot(5,5,idx+1)  
                plt.imshow(W[:,idx].reshape((60,60)),cmap='gray')#W is composing of 25 largest eigenvector of cov"S", this will show the eigenface
                plt.axis('off')
        plt.savefig(f'./{folder}/{title}/{title}.png')
    # for i in range(W.shape[1]):
    #     plt.clf()
    #     plt.title(f'{title}_{i+1}')
    #     plt.imshow(W[:,i].reshape((60,60)),cmap='gray')
    #     plt.savefig(f'./{folder}/{title}/{title}.png')
    
    #show reconstruction of random 10 images
    if reconstruct.shape[0] == 10:
        plt.clf()
        for i in range(2):
            for j in range(5):
                idx = i*5+j
                plt.subplot(2,5,idx+1)
                plt.imshow(reconstruct[idx].reshape((60,60)),cmap='gray')#W is composing of 25 largest eigenvector of cov"S", this will show the eigenface
                plt.axis('off')
        plt.savefig(f'./{folder}/{title}/10_reconstructions.png')  

    #reconstruct and store the images one by one instead
    #for i in range(reconstruct.shape[0]):
    # plt.clf()    
    # plt.title(picked_fn[i])
    # plt.imshow(reconstruct[i].reshape((60,60)),cmap='gray')
    # plt.savefig(f'./{folder}/{title}/{picked_fn[i]}.png')
    
        


