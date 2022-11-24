import os
from math import factorial

if __name__ == '__main__':
    path = input('file path: ')
    name = input('file name: ')
    filepath = os.path.join(path,name)
    fp=open(filepath,'r')
    line=fp.readline()
    
    #set initial parameters for beta prior
    a = int(input('input a:'))
    b = int(input('input b:'))

    case = 1

    while line:
        #we implement the final case: beta-binomial conjugate
        #p(theta| x) -> beta distribution
        
        pos = line.count('1') #count positive cases
        neg = line.count('0')
        sum = pos+neg
        #print(pos , neg)
        #positive probability
        pp = pos/sum
        #negative prob
        np = neg/sum
        #below is the beta distribution posterior, where online learning will treat this as new prior for the next case posterior
        #likelihood
        #print(pos,neg,sum)
        LL = factorial(sum)/factorial(pos)/factorial(neg)*pp**pos*np**neg

        print('case{}:{}'.format(case,line))
        print('likelihood:{}'.format(LL))
        print('beta prior: a:{}, b:{}'.format(a,b))
        a+=pos #adding the parameter we know in current case and move on to the next case
        b+=neg
        print('beta posterior: a:{}, b:{}'.format(a,b))
        print()
        print('------------------------------')
        line=fp.readline()
        case+=1
