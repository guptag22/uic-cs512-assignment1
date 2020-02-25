# Imports
import numpy as np
import scipy.optimize as opt
import string
import math

input_path = "/Users/wangfei/Documents/Courses/CS/CS512/HW/HW1/Assignment_1_Programming/data/"

decode_input = np.genfromtxt(input_path + "decode_input.txt", delimiter = ' ')

X = decode_input[:100*128]
W = decode_input[100*128:100*128+26*128]
T = decode_input[100*128+26*128:]


def decode(X, W, T):
    W = np.reshape(W, (26, 128))  # each row of W is w_y (128 dim)
    T = np.reshape(T, (26, 26))   # T is 26*26
    T = T.transpose()             # To make T11, T21, T31 ... T26,1 the first items of the rows 
    m = int(len(X)/128)           # length of the word
    X = np.reshape(X, (m, 128))
    # Node potential: f_ys = <W_ys=i, xs>; Edge potential: g_ys,ys+1 = Tys, ys+1 
    C = np.zeros([26, 26])        # C is a 26 * 26 matrix storing C11 = f_a+g_aa, C12 = f_a+g_ab, ... C26,26 = f_z+g_zz
    l = 0                    # C_max is the maximum summation of f_ys and g_ys,ys+1
    ls = []
    ls.append(l)
    y_star = np.zeros([m, 1], dtype=np.int8)
    obj = np.zeros([26,1])
    lm = np.zeros([26,1])
    for s in range(m-1):
        for i in range(26):
            for j in range(26):
                C[i, j] = np.dot(W[i, :], X[s, :]) + T[i, j] + l
        l = np.amax(C)
        ls.append(l)
    for i in range(26):
        lm[i] = np.dot(W[i, :], X[m-1, :]) + ls[m-2]
    y_star[m-1] = np.argmax(lm)
    for s in range(m-2, -1, -1):
        for i in range(26):
            obj[i] = np.dot(W[i, :], X[s, :]) + T[i, y_star[s+1]] + ls[s]
        y_star[s] = np.argmax(obj)
    #return y_star;
    node = 0
    edge = 0
    for j in range(m-1):
        yj_star = y_star[j]
        yj1_star = y_star[j+1]
        node += np.dot(W[yj_star, :], X[j, :]) 
        edge += T[yj_star, yj1_star]
    obj = node + edge + np.dot(W[y_star[m-1], :], X[m-1, :])

    return y_star+1, obj;






