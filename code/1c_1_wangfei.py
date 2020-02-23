# Imports
import numpy as np
import scipy.optimize as opt
import string

# get the index of letters a-z
# y_conv = dict(enumerate(string.ascii_lowercase, 1)) 
# def get_key(val):
#     for key, value in y_conv.items():
#         if str(val) == value: 
#             return key;

input_path = "/Users/wangfei/Documents/Courses/CS/CS512/HW/HW1/Assignment_1_Programming/data/"

decode_input = np.genfromtxt(input_path + "decode_input.txt", delimiter = ' ')

X_decode = decode_input[:100*128]
W_decode = decode_input[100*128:100*128+26*128]
T_decode = decode_input[100*128+26*128:]


def decode(X, W, T):
    W = np.reshape(W, (26, 128))  # each row of W is w_y (128 dim)
    T = np.reshape(T, (26, 26))   # T is 26*26
    T = T.transpose()             # To make T11, T21, T31 ... T26,1 the first items of the rows 
    m = int(len(X)/128)           # length of the word
    X = np.reshape(X, (m, 128))
    # Node potential: f_ys = <W_ys=i, xs>; Edge potential: g_ys,ys+1 = Tys, ys+1 
    C = np.empty([26, 26])       # C is a 26 * 26 matrix storing C11 = f_a+g_aa, C12 = f_a+g_ab, ... C26,26 = f_z+g_zz
    C_max = 0                    # C_max is the maximum summation of f_ys and g_ys,ys+1
    y_star = np.empty([m, 1], dtype=np.int8)
    obj = np.empty([26, 1])      # obj is the objective function that we want to maximize in equation(3)
    C_m = np.empty([26, 1])      # C_m is the C_max at the last node. initialize first with a 26-vector
    C_max_vector = []            # C_max_vector: a vector that stores all C_max values
    obj_vector = []
    for s in range(m-1):
        for i in range(26):       #yi takes a, b, ... z
            for j in range(26):   #yi+1 takes a, b, ... z
                C[i, j] = np.add(np.add(np.dot(W[i, :], X[s, :]), T[i, j]), C_max) 
        C_max = np.amax(C)
        C_max_vector.append(C_max)
    # for the last node, it does not have a letter next to it, so the edge potential vanishes. 
    # Only taking into account the node weight.
    for i in range(26):
        C_m[i] = np.add(C_max, np.dot(W[i, :], X[m-1, :]))  
    C_max_m = np.amax(C_m)            # maximum value of C_m = C_ym + f_ym
    y_star[m-1] = np.argmax(C_m)      #ym*
    C_max_vector.append(C_max_m)
    obj_vector.append(C_max_m)
    # from ym*, to find ym-1*, then ym-2* ... y2* using the calculated C_max's from previous steps
    # Previously, cy2, cy3, ... cym were stored in C_max_vector from index 0 to m-2. 
    # The last entry in C_max_vector is max_ym{C_ym+f_ym} (taking index m-1)
    for s in range(m-2, 0, -1):  
        for i in range(26):
            # plug in the y*_m to find ym-1* ... 
            obj[i] = np.add(np.add(np.dot(W[i, :], X[s, :]), C_max_vector[s+1]), T[i, y_star[s+1]]) 
        obj_max = np.amax(obj)
        obj_vector.append(obj_max)
        y_star[s] = np.argmax(obj)
    # y1* = argmax(f_y1+g_y1y2*)
    for i in range(26):
        obj[i] = np.add(np.dot(W[i, :], X[0, :]), T[i, y_star[1]]) # fy1
    obj_max = np.amax(obj)
    obj_vector.append(obj_max)
    y_star[0] = np.argmax(obj)
    return(np.add(y_star,1)); # add one index to y* to make 1 -> a, 2 -> b ... instead of 0 -> a, 1 -> b ... 

y_predict = decode(X_decode, W_decode, T_decode)

#test the maximum objective value 
m = int(len(X_decode)/128)
edge = 0
node = 0
W_decode = np.reshape(W_decode, (26, 128))  # each row of W is w_y (128 dim)
T_decode = np.reshape(T_decode, (26, 26))   # T is 26*26
T_decode = T_decode.transpose() # To make T11, T21, T31 ... T26,1 the first items of the rows 
m = int(len(X_decode)/128)
X_decode = np.reshape(X_decode, (m, 128))
for j in range(m-1): 
    ys = y_predict[j, 0] - 1   # convert the index back to 0, 1, 2 (a, b, c...) instead of 1, 2, 3
    ys_p_1 = y_predict[j+1, 0] - 1
    node += np.dot(W_decode[ys, :], X_decode[j, :])
    edge += T_decode[ys, ys_p_1]

idx_m = y_predict[-1, 0] - 1
node + edge + np.dot(W_decode[idx_m, :], X_decode[m-1, :])







