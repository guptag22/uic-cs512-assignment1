# Imports
import os
import numpy as np
import scipy.optimize as opt
import string
from numpy.linalg import norm

train_data_PATH = "/Users/gammu/Documents/AML_assignment_1/uic-cs512-assignment1/data/train.txt"
file = open(train_data_PATH,"r")
train_data = (file.read()).strip()
train_data = train_data.split('\n')

test_data_PATH = "/Users/gammu/Documents/AML_assignment_1/uic-cs512-assignment1/data/test.txt"
file = open(test_data_PATH,"r")
test_data = (file.read()).strip()
test_data = test_data.split('\n')

model_PATH = "/Users/gammu/Documents/AML_assignment_1/uic-cs512-assignment1/data/model.txt"
model = np.genfromtxt(model_PATH, delimiter = ' ')

def get_lhood_grad(Yt,Xt,W,T) :         ## get log(p(Yt|Xt)) and gradients for one training example
    m = len(Yt)                         ## number of letters in the word

    a = np.zeros((m,26))                ## message forward
    b = np.zeros((m,26))                ## message backwards
    
    def f(s,y) :                        ## < Wy, Xts >
        return np.dot(W[y,:],Xt[s,:])   
    def g(i,j) :                        ## T[i,j]
        return T[i,j]                   

    for s in range(m-2,-1,-1) :         ## Calculate backward messages b
        for j in range(26) :            ## s goes from m-2 to 0. b[m-1,i] = 1 for all i from 0 to 25
            res = []
            for i in range(26) :
                res.append(f(s+1,i) + g(j,i) + b[s+1,i])
            max_value = max(res)
            res = max_value + np.log(sum(np.exp(np.array(res) - max_value)))
            b[s,j] = res

    for s in range(1,m,1) :             ## Calculate forward messages a
        for j in range(26) :
            res = []
            for i in range(26) :
                res.append(f(s-1,i) + g(i,j) + a[s-1,i])
            max_value = max(res)
            res = max_value + np.log(sum(np.exp(np.array(res) - max_value)))
            a[s,j] = res
            
    log_Zx = 0                                      ## Zx sum over all yi for given training example
    res = []
    for i in range(26) :
        res.append(f(0,i) + b[0,i])   
    max_value = max(res)
    res = max_value + np.log(sum(np.exp(np.array(res) - max_value)))
    log_Zx = res       

    ## Calculate marginals
    def marginal_ys(s,ys) :
        return (np.exp(f(s,ys) + a[s,ys] + b[s,ys] - log_Zx))
    def marginal_ys_ys1(s,ys,ys1) :
        return (np.exp(f(s,ys) + f(s+1,ys1) + g(ys,ys1) + a[s,ys] + b[s+1,ys1] - log_Zx)

    ## Calculate log(p(Y|X))
    log_p_Yt = (f(m-1,Yt[m-1]) - log_Zx)
    for s in range(m-1) :
        log_p_Yt += f(s,Yt[s]) + g(Yt[s-1],Yt[s])   ## log(p(Yt|Xt)) = sum_s (f_s(y_s)) + sum_s(g(y_s-1,y_s) - log(Zx))

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
            grad_Tij_t[i,j] = res
    
    return log_p_Yt, grad_Wy_t, grad_Tij_t

###################### Test code ######################
word_list = train_data[:24]
Y = []              ## array of all labels of all words
X = []              ## array of pixel values of all words
Yt = []              ## array of labels of letters for a single word
Xt = []              ## array of pixel values of letters for a single word
for i in range(len(word_list)) :
    letter = (word_list[i]).split()        ## extract data corresponding to a single letter
    Yt.append(ord(letter[1].lower())-97)     ## extract the label and convert it to integer
    Xt.append(np.array(list(map(int,letter[5:]))))       ## extract pixel values, convert them to integer 
                                                        ## and form a numpy array for data manipulation
    if (int(letter[2]) == -1) :             ## Check for end of word
        Y.append(Yt)
        X.append(np.array(Xt))
        Yt = []
        Xt = []
W = model[:128*26]
T = model[128*26:]
W = np.reshape(W, (26,128))                 ## W contains each Wy as a row of size 128
T = np.reshape(T, (26,26))
T = T.transpose()

log_p_Y1, grad_Wy_1, grad_Tij_1 = get_lhood_grad(Y[-1],X[-1],W,T)

print(log_p_Y1, grad_Wy_1, grad_Tij_1)

######################################################

# def crf_obj(model,word_list,C) : 
#     W = model[:128*26]
#     T = model[128*26:]
#     W = np.reshape(W, (26,128))                 ## W contains each Wy as a row of size 128
#     T = np.reshape(T, (26,26))
#     T = T.transpose()
#     N = int((word_list[-1].split())[3])         ## Total number of words in the training data
#     ##### Extract Y and X from data ######
#     ##### shape of X should be each letter in a row of size 128
#     Y = []              ## array of all labels of all words
#     X = []              ## array of pixel values of all words
#     Yt = []             ## array of labels of letters for a single word
#     Xt = []             ## array of pixel values of letters for a single word
#     for i in range(len(word_list)) :
#         letter = (word_list[i]).split()         ## extract data corresponding to a single letter
#         Yt.append(ord(letter[1].lower())-97)    ## extract the label and convert it to integer
#         Xt.append(np.array(list(map(int,letter[5:]))))      ## extract pixel values, convert them to integer 
#                                                             ## and form a numpy array for data manipulation
#         if (int(letter[2]) == -1) :             ## Check for end of word
#             Y.append(Yt)
#             X.append(np.array(Xt))
#             Yt = []
#             Xt = []
    
#     ## Calculate gradient of whole training data
#     grad_Wy = np.zeros((26,128))
#     grad_T = np.zeros((26,26))
#     log_liklihood = 0
#     for t in range(N) :
#         log_p_Yt, grad_Wy_t, grad_Tij_t = get_lhood_grad(Y[t], X[t], W, T)
#         log_liklihood += log_p_Yt / N
#         grad_Wy += grad_Wy_t / N
#         grad_T += grad_Tij_t / N
#     obj = ((norm(W)**2 + norm(T)**2) /2) - C * log_liklihood
#     ## flatten grad_Wy and grad_T and concatenate
#     flat_grad_Wy = []
#     flat_grad_T = []
#     for i in range(26) :
#         flat_grad_Wy.append(grad_Wy[i,:])
#         flat_grad_T.append(grad_T[:,i])
#     grad_theta = flat_grad_Wy + flat_grad_T         ## Gradient vector
#     return [obj, grad_theta]

# def crf_test(model, word_list) :
#     W = model[:128*26]
#     T = model[128*26:]
#     W = np.reshape(W, (26,128))                 ## W contains each Wy as a row of size 128
#     T = np.reshape(T, (26,26))
#     T = T.transpose()
#     # N = int((word_list[-1].split())[3]) 
#     ##### Extract Y and X from data ######
#     ##### shape of X should be each letter in a row of size 128
#     Y = []              ## array of all labels of all words
#     X = []              ## array of pixel values of all words
#     Yt = []             ## array of labels of letters for a single word
#     Xt = []             ## array of pixel values of letters for a single word
#     for i in range(len(word_list)) :
#         letter = (word_list[i]).split()         ## extract data corresponding to a single letter
#         Yt.append(ord(letter[1].lower())-97)    ## extract the label and convert it to integer
#         Xt.append(np.array(list(map(int,letter[5:]))))      ## extract pixel values, convert them to integer 
#                                                             ## and form a numpy array for data manipulation
#         if (int(letter[2]) == -1) :             ## Check for end of word
#             Y.append(Yt)
#             X.append(np.array(Xt))
#             Yt = []
#             Xt = []
#     y_predict = crf_decode(W, T, word_list)     ## Decode utility defined in 1c

#     ## Calculate accuracy by comparing y_pred with true labels
#     ##----------- TO-DO -----------##
#     return accuracy

# def optimize_obj(train_data, test_data, C) :
#     Wo = np.zeros((128*26+26**2,1))
#     result = opt.fmin_tnc(crf_obj, Wo, args = [train_data, C], maxfun=100,
#                           ftol=1e-3, disp=5)
#     model = result[0]
#     accuracy = crf_test(model, test_data)
#     print('CRF test accuracy for c = {}: {}'.format(C,accuracy))
#     return accuracy