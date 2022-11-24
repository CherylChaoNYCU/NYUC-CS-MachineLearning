import numpy as np
import matplotlib.pyplot as plt

def plot_img(p_img): #the input is one row of pattern matrix of specific class, so the shape is 1x784
    for i in range(28):
        for j in range(28):
            print(p_img[i*28+j],end=' ')
        print()
    print()
    print()
    return


#match_class:0~9 totoal 10rows:ex = [1,4,7,0,3...]means that when printing class0, we need to output the pattern "P1" from the predicted distribution
def create_pattern(Pred,match_class,th):

    img_pattern = np.asarray(Pred > th,dtype='uint8') #>threshold: print1,else print 0,dimension:10*784
    #print(img_pattern)
    for i in range(10):
        print('class {}:'.format(i))
        plot_img(img_pattern[match_class[i]])
    return



def compose_confusion(gt,pred_class,matched_class):

 
    for  i in range(10): #pair(0,0)  / (0,1) ... 
        c = matched_class[i]
        tp = fp = fn = tn = 0
        for i in range(60000):
            if gt[i] != c and pred_class[i]!=c:
                tn+=1
            elif gt[i] == c and pred_class[i]==c:
                tp+=1
            elif gt[i] == c and pred_class[i]!=c:
                fn+=1
            else:
                fp+=1
        print_confusion(c,tn,tp,fn,fp)
      

def print_confusion(c,tn,tp,fn,fp):
    print('Confusion Matrix {}:'.format(c))
    print('            Predict number {}  Predict not number{}'.format(c,c))
    print('Is number {}         {}             {}    '.format(c,tp,fn))
    print('Is not number{}      {}             {}    '.format(c,fp,tn))

    print('\n')
    print('Sensitivity (Successfully predict number {}):{}'.format(c,tp/(tp+fn)))
    print('Specificity (Successfully predict not number {}):{}'.format(c,fp/(fp+tn)))
    print()

def print_error_iter(steps,gt,pred_class,matched_class):

    #we want to find the difference between matched class and predicted class with max posterior wi
    #read supposed match class of each input image
    supposed_class = np.zeros(60000)
    for i in range(60000):
        supposed_class[i] = matched_class[gt[i]]
    error = np.count_nonzero(supposed_class-pred_class)
    rate = error/60000
    print('Total iteration to converge: {}'.format(steps))
    print('Total error rate:{}'.format(rate))


