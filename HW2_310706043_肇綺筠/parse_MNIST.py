
'''The database contains 70,000 28x28 black and white images representing the digits 
zero through nine. The data is split into two subsets, with 60,000 images belonging to 
the training set and 10,000 images belonging to the testing set. 
'''
'''
about test sets:
The first 5000 examples of the test set are taken from the original MNIST training set. 
The last 5000 are taken from the original MNIST test set. The first 5000 are cleaner and easier than the last 5000.


'''
from sys import byteorder
import numpy as np

def parse():
    train_x = open('train-images-idx3-ubyte','rb')
    train_y = open('train-labels-idx1-ubyte','rb') #labeling images
    test_x=open('t10k-images-idx3-ubyte','rb')
    test_y=open('t10k-labels-idx1-ubyte','rb')

    img_size = 28
    img_num = 60000

    #read MNIST

    #READ TRAIN SETS
    train_x.read(16) #discard header infos
    train_y.read(8)
    #60000 imgs (rows), each img has 28*28 cols (flatten to 1*28*28, and total is 60000 pieces)
    train_x_buf = np.zeros((60000,28*28),dtype='uint8')
    train_y_buf = np.zeros(60000,dtype='uint8') #labels:60000
    #transfering each pixel from big endian form to decimal and store it in buffers
    for i in range(60000):
        for j in range(28*28):
            train_x_buf[i,j] = int.from_bytes(train_x.read(1),byteorder = 'big') #transfering image pixels
        train_y_buf[i] = int.from_bytes(train_y.read(1),byteorder = 'big') #reading labels


    #READ TEST SETS
    test_x.read(16) 
    test_y.read(8)
    #10000 imgs for testing
    test_x_buf = np.zeros((10000,28*28),dtype='uint8')
    test_y_buf = np.zeros(10000,dtype='uint8') #labels:60000
    #transfering each pixel from big endian form to decimal and store it in buffers
    for i in range(10000):
        for j in range(28*28):
            test_x_buf[i,j] = int.from_bytes(test_x.read(1),byteorder = 'big') #transfering image pixels
        test_y_buf[i] = int.from_bytes(test_y.read(1),byteorder = 'big') #reading labels

    return (train_x_buf,train_y_buf),(test_x_buf,test_y_buf)


        

        



