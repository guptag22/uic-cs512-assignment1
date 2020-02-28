import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
# Data
x= ['1','10','100','1000','5000']
y1 = [16.7,26.3,41.35,48.3,49.5]
y2 = [47.65,60.45,67.89,69.7,69.9557]
y3 = [30,40,60,70,80]

w = 10
h = 8
d = 100
plt.figure(figsize=(w, h), dpi=d)

plt.plot(x, y1,marker='', color='olive', linewidth=2, linestyle='dashed', label="SVM_HMM")
plt.plot(x, y2, color='red', linewidth=2, linestyle='dashed', label="SVM_MC")
plt.plot(x, y3, color='green', linewidth=3,  label="CRF")
plt.legend()

plt.savefig("letteraccuracies.png")

y1 = [1.6,7.2,14.8,16.7,17]
y2 = [16.66,26.22,41.3,48.2,49.4]
y3 = [30,40,60,70,80]

plt.figure(figsize=(w, h), dpi=d)

plt.plot(x, y1,marker='', color='olive', linewidth=2, linestyle='dashed', label="SVM_HMM")
plt.plot(x, y2, color='red', linewidth=2, linestyle='dashed', label="SVM_MC")
plt.plot(x, y3, color='green', linewidth=3,  label="CRF")
plt.legend()

plt.savefig("wordaccuracies.png")


