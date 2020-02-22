# Imports
import numpy as np
import scipy.optimize as opt
import string

# get the index of letters a-z
y_conv = dict(enumerate(string.ascii_lowercase, 1)) 
def get_key(val):
    for key, value in y_conv.items():
        if str(val) == value: 
            return key;

input_path = "/Users/wangfei/Documents/Courses/CS/CS512/HW/HW1/Assignment_1_Programming/data/"

decode_input = np.genfromtxt(input_path + "decode_input.txt", delimiter = ' ')

X_decode = decode_input[:100*128]
W_decode = decode_input[100*128:100*128+26*128]
T_decode = decode_input[100*128+26*128:]

def decode(X, W, T):
    W = np.reshape(W, (128, 26))  # each column of W is w_y (128 dim)
    T = np.reshape(T, (26, 26))   # T is 26*26
    m = int(len(X)/128)
    X = np.reshape(X, (m, 128))
    C = np.empty([26, 26])
    C_max = 1
    y_star = np.empty([m, 1])
    for s in range(m):
        for i in range(26):       #yi takes a, b, ... z
            for j in range(26):   #yi+1 takes a, b, ... z
                C[i, j] = np.multiply(np.multiply(np.dot(W[:, i], X[s, :]), T[i, j]), C_max)
        C_max = np.amax(C)
        C_max_coordinate = np.where(C == C_max)
        i_max = C_max_coordinate[0]   # the argmax of Cy_{s+1}  the corresponding i will be the letter that maximize Cy_{s+1}
        j_max = C_max_coordinate[1]
        y_star[s] = i_max + 1 #to make 'a' == 1 instead of 'a' == 0
    return(y_star);

y_predict = decode(X_decode, W_decode, T_decode)

print(y_predict)