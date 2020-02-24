import numpy as np

loaded_arr = np.loadtxt('../data/decode_input.txt')


def max_sum(X, W, T):
    l = list()
    l.append(0)
    for i in range(1, X.shape[0]):
        temp_list = list()
        for y in range(0, 26):
            w_x = np.dot(W[y], X[i])
            for y_next in range(0, 26):
                w_x_t = w_x + T[y][y_next]
                w_x_t = w_x_t + l[i-1]
                temp_list.append(w_x_t)
        # print(np.amax(temp_list), l[i-1])
        l.append(np.amax(temp_list))
    return(np.array(l))


# computing separate l's
# def max_sum(X, W, T):
#     all_ls = list()
#     for i in range(0, X.shape[0]):
#         l = list()
#         l.append(0)
#         temp_list = list()
#         for y in range(1, 26):
#             w_x = np.dot(W[y], X[i])
#             for y_next in range(0, 26):
#                 w_x_t = w_x + T[y][y_next]
#                 w_x_t = w_x_t + l[y-1]
#                 temp_list.append(w_x_t)
#         # print(np.amax(temp_list), l[i-1])
#             l.append(np.amax(temp_list))
#         all_ls.append(l)
#         # print(l)
#     return(np.array(all_ls))


def get_argmax(X, W, T, l):
    pred = list()
    y_m = W.shape[0]-1
    m = X.shape[0]-1
    # print(y_m, m)
    # print(l.shape)
    # print(W[y_m].shape, X[m].shape)
    y_star = np.dot(W[y_m], X[m]) + l[m]
    pred.append(np.argmax(y_star) + 1)
    # pred.append(np.dot(W[y_m], X[m]) + l[m][y_m])
    
    for i in range(m-1, -1, -1):
        argmax_list = list()
        for y in range(0, 26):
            dot_prod = np.dot(W[y], X[i])
            dot_prod = dot_prod + T[y][pred[m-i-1]-1] + l[i]
            argmax_list.append(dot_prod)
        pred.append(np.argmax(argmax_list) + 1)
        # print(i, len(pred), pred[m-i-1])
    return np.array(pred)


X = loaded_arr[:100*128]
# print(X.shape)
X = np.reshape(X, (100, 128))
print(X.shape)


W = loaded_arr[100*128:(100*128)+(26*128)]
# print(W.shape)
W = np.reshape(W, (26, 128))
print(W.shape)


T = loaded_arr[(100*128)+(26*128):]
# print(T.shape)
T = np.reshape(T, (26, 26))
print(T.shape)

a = np.dot(X[1], W[1])
b = a + T[0][1]
c = np.exp(b)
# print(c)
# print(np.amax([c, 1]))


l = max_sum(X, W, T)
print('l shape:', l.shape)
pred = get_argmax(X, W, T, l)
print(pred)
