# Imports
import os
import numpy as np
import scipy.optimize as opt
import string
from numpy.linalg import norm

train_data_PATH = os.path.join(os.path.dirname(os.getcwd()), "data", "train.txt")
train_data = np.genfromtxt(train_data_PATH, delimiter = ' ')

test_data_PATH = os.path.join(os.path.dirname(os.getcwd()), "data", "test.txt")
test_data = np.genfromtxt(test_data_PATH, delimiter = ' ')

model_PATH = os.path.join(os.path.dirname(os.getcwd()), "data", "model.txt")
model = np.genfromtxt(model_PATH, delimiter = ' ')

def get_lhood_grad(Yt,Xt,W,T) :         ## get log(p(Yt|Xt)) and gradients for one training example
    m = len(Yt)                         ## number of letters in the word

    a = np.ones((m,26))                 ## message forward
    b = np.ones((m,26))                 ## message backwards
    
    def f(s,y) :
        return np.dot(W[y,:],Xt[s,:])   ## np.exp(np.dot(W[y,:],X[s,:]))
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
    log_p_Yt = f(0,Yt[0]) - np.log(Zx)
    for i in range(1,m,1) :
        log_p_Yt += f(i,Yt[i]) + g(Yt[i-1],Yt[i])   ## log(p(Yt|Xt)) = sum_s (f_s(y_s)) + sum_s(g(y_s-1,y_s) - log(Zx))

    ## Calculate Gradient wrt Wy of log(p(Y|X))
    grad_Wy_t = np.empty((26,128))
    for i in range(26) :
        res = 0
        for s in range(m) :
            ind = 1 if Yt[s] == i else 0
            res += (ind - marginal_ys(s,i)) * Xt[s,:]
        grad_Wy_t[i,:] = res

    ## Calculate Gradient wrt Tij of log(p(Y|X))
    grad_Tij_t = np.empty((26,26))
    for i in range(26) :
        for j in range(26) :
            res = 0
            for s in range(m-1) :
                ind = 1 if (Yt[s] == i and Yt[s+1] == j) else 0
                res += ind - marginal_ys_ys1(s,Yt[s],Yt[s+1])
            grad_Tij_t = res
    
    return log_p_Yt, grad_Wy_t, grad_Tij_t

def crf_obj(model,word_list,C) : 
    W = model[:128*26]
    T = model[128*26:]
    W = np.reshape(W, (26,128))                 ## W contains each Wy as a row of size 128
    T = np.reshape(T, (26,26))
    T = T.transpose()
    
    ##### Extract Y and X from data ######
    ##### shape of X should be each letter in a row of size 128
    ##----------- TO-DO -----------##
    # Calculate gradient of whole training data
    # Define N = Total number of words in the training data
    ##----------- TO-DO -----------##
    grad_Wy = np.empty((26,128))
    grad_T = np.empty((26,26))
    log_liklihood = 0
    for t in range(N) :
        log_p_Yt, grad_Wy_t, grad_Tij_t = get_lhood_grad(Y[t], X[t], W, T)
        log_liklihood += log_p_Yt / N
        grad_Wy += grad_Wy_t / N
        grad_T += grad_Tij_t / N
    obj = ((norm(W)**2 + norm(T)**2) /2) - C * log_liklihood
    ## flatten grad_Wy and grad_T and concatenate
    ##----------- TO-DO -----------##
    grad_theta = flat_grad_Wy + flat_grad_T
    return [obj, grad_theta]

def crf_test(model, word_list) :
    W = model[:128*26]
    T = model[128*26:]
    W = np.reshape(W, (26,128))                 ## W contains each Wy as a row of size 128
    T = np.reshape(T, (26,26))
    T = T.transpose()
    ##### Extract Y and X from data ######
    ##### shape of X should be each letter in a row of size 128
    ##----------- TO-DO -----------##
    y_predict = crf_decode(W, T, word_list)     ## Decode utility defined in 1c

    ## Calculate accuracy by comparing y_pred with true labels
    ##----------- TO-DO -----------##
    return accuracy


def optimize_obj(train_data, test_data, C) :
    Wo = np.zeros((128*26+26**2,1))
    result = opt.fmin_tnc(crf_obj, Wo, args = [train_data, C], maxfun=100,
                          ftol=1e-3, disp=5)
    model = result[0]
    accuracy = crf_test(model, test_data)
    print('CRF test accuracy for c = {}: {}'.format(C,accuracy))
    return accuracy