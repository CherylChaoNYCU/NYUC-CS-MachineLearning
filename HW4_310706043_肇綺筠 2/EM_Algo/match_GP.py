import numpy as np
from scipy.optimize import linear_sum_assignment

'''in this part, we want to minimize the total distance of real 
and predicted distribution.

I used 匈牙利Algo:https://blog.csdn.net/your_answer/article/details/79160045
to find the best match(reorder the predicted class that minimize distribution distance
'''

def distance(g,p):
    return np.linalg.norm(g-p)

def H_algo(cost):
    
    #this returns the column index of the element that each row should choose to minimize the total cost
    r_id,c_id = linear_sum_assignment(cost)
    return c_id #this returns the actual class that predicted distribution should really assign
    


def match(gt,pred):

    cost_matrix = np.zeros((10,10)) #this stores the distance of (class i ground truth) with class0~9 predicted distribution

    for i in range(10):
        for j in range(10):
            cost_matrix[i,j] = distance(gt[i],pred[j])
    new_class = H_algo(cost_matrix)
    return new_class

