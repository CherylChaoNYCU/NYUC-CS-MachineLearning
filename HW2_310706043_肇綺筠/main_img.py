import sys
from parse_MNIST import parse
from functions.discrete import *
from functions.continuous import *
import numpy as np
import math

#calculate the prior of each class (p(c1)...) => bayesian needed in nominator

def class_prior(train_y):

    pri = np.zeros(10) #class: 0-9
    for c in range(10):
        pri[c] = np.sum(train_y == c) / len(train_y) #class is the index, we need to calculate prior of each class(ex. 2000 zeros in 60000 labels => p(0) = 5000/60000)
    return pri

def draw_num(pixel_prob,th):
    print('Imagination of numbers in Bayesian classifier: ')
    #print(pixel_prob)
    for class_ in range(10):
        print('{}:'.format(class_))
        #pixels:28*28
        for i in range(28):
            for j in range(28):
                print('1' if np.argmax(pixel_prob[class_,i*28+j])>=th else '0',end=' ')
            print()
        print()
    print()




if __name__ == '__main__':
    (tn_x,tn_y),(tt_x,tt_y) = parse()

    mode = input('toggle option (0:discrete / 1:continuous): ')

    if mode == '0':
        pixel_prob = discrete_prob(tn_x,tn_y) #get the pixel prob of each image in train sets
        prior = class_prior(tn_y)
        #testing
        discrete_test(len(tt_y),pixel_prob,prior,tt_x,tt_y)
        draw_num(pixel_prob,16) #discrete form transfers pixel values into bins (//8), so the threshold should be 32/2 = 16 (>0.5 will show 1)
    else:

        ev = 10 #epsilon of variance
        ep = 1e-30 #avoid 0 variance in continuous mode, epsilon of probability
        pixel_prob = cont_prob(tn_x,tn_y,ev) #get the pixel prob of each image in train sets
        prior = class_prior(tn_y)
        #testing
        cont_test(len(tt_y),pixel_prob,prior,tt_x,tt_y,ep)
        draw_num(pixel_prob,128) #discrete form transfers pixel values into bins (//8), so the threshold should be 128/8 = 16






