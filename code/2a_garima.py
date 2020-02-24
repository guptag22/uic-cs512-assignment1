# Imports
import os
import numpy as np
import scipy.optimize as opt
import string

train_data_PATH = os.path.join(os.path.dirname(os.getcwd()), "data", "train.txt")
train_data = np.genfromtxt(train_data_PATH, delimiter = ' ')

model_PATH = os.path.join(os.path.dirname(os.getcwd()), "data", "model.txt")
model = np.genfromtxt(model_PATH, delimiter = ' ')
W = model[:128*26]
T = model[128*26:]

W = np.reshape(W, (26,128))
T = np.reshape(T, (26,26))
T = T.transpose()

######### extract X and Y #########

def get_log_lhood_gradient(Y,X,W,T) :
    m = len(Y)              ## number of letters in the word

    a = np.ones((m,26))     ## message forward
    b = np.ones((m,26))     ## message backwards
    
    def f(s,y) :
        return np.dot(W[y,:],X[s,:])    ## np.exp(np.dot(W[y,:],X[s,:]))
    def g(i,j) :
        return T[i,j]                   ## np.exp(T[i,j])

    def b_rec(s,y) :
        if (s < m-1) :
            res = 0
            for i in range(26) :
                res += np.exp(f(s+1,i) + g(y,i) + b_rec(s+1,i))
            b[s,y] = res
        return np.log(b[s,y])
    def a_rec(s,y) :
        if (s > 0) :
            res = 0
            for i in range(26) :
                res += np.exp(f(s-1,i) + g(i,y) + a_rec(s-1,i))
            a[s,y] = res
        return np.log(a[s,y])

    Zx = 0
    for i in range(26) :
        Zx += np.exp(f(0,i)) * b[0,i]           ## all values of b are calculated and stored 

    for j in range(26) :
        a[m-1,j] = np.exp(a_rec(m-1,j))         ## all values of a are calculated and stored
    
    ## Calculate marginals
    def marginal_ys(s,ys) :
        return (np.exp(f(s,ys)) * a[s,ys] * b[s,ys]) / Zx
    def marginal_ys_ys1(s,ys,ys1) :
        return (np.exp(f(s,ys) + f(s+1,ys1) + g(ys,ys1)) * a[s,ys] * b[s+1,ys1]) / Zx

    ## Calculate log(p(Y|X))
    log_p_y = f(0,Y[0]) - np.log(Zx)
    for i in range(1,m,1) :
        log_p_y += f(i,Y[i]) + g(Y[i-1],Y[i])   ## log(p(Y|X)) = sum_s (f_s(y_s)) + sum_s(g(y_s-1,y_s) - log(Zx))

    ## Calculate Gradient wrt Wy of log(p(Y|X))
    grad_Wy = np.empty((26,128))
    for i in range(26) :
        res = 0
        for s in range(m) :
            ind = 1 if Y[s] == i else 0
            res += (ind - marginal_ys(s,i)) * X[s]
        grad_Wy[i,:] = res

    ## Calculate Gradient wrt Tij of log(p(Y|X))
    grad_Tij = np.empty((26,26))
    for i in range(26) :
        for j in range(26) :
            res = 0
            for s in range(m-1) :
                ind = 1 if (Y[s] == i and Y[s+1] == j) else 0
                res += ind - marginal_ys_ys1(s,Y[s],Y[s+1])
            grad_Tij = res
    
    return log_p_y, grad_Wy, grad_Tij

    