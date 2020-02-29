import cv2
import numpy as np
import matplotlib.pyplot as plt

ft = open("transform.txt")
tranlist=[]
for line in ft:
    line = line.split()
    tranlist.append(line)
    
ft.close()

lindex = {}
alphabets= ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
digits = {'0':0,'1':1}
for i in range(len(alphabets)):
        lindex[alphabets[i]] = i+1
ftrain = open('train.txt')
xid,xletter,xnextid,xwordid,xpixels,xposition = [],[],[],[],[],[]
for line in ftrain:
    fields=line.split(" ")
    fields[-1] = fields[-1].strip()
    xid.append(fields[0])
    xletter.append(lindex[fields[1]])
    xnextid.append(fields[2])
    xwordid.append(fields[3])
    xposition.append(fields[4])
    temp=[]
    for i in range(len(fields[5:])):
        temp.append(digits[fields[5+i]])
    xpixels.append(temp)
ftrain.close()  




