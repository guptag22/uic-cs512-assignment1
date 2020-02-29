import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
# Data
x= ['0','500','1000','1500','2000']
y1 = [69.95,69.42,69.05,68.6,68.15]
y2 = [30,40,60,70,80]

w = 10
h = 8
d = 100
plt.figure(figsize=(w, h), dpi=d)

plt.plot(x, y1,marker='', color='olive', linewidth=2, linestyle='dashed', label="SVM_MC")
plt.plot(x, y2, color='green', linewidth=3,  label="CRF")
plt.legend()

plt.savefig("letteraccuracies.png")

y1 = [17.2,16.75,16.3,15.3,15.09] 
y2 = [60,60,60,60,60]

plt.figure(figsize=(w, h), dpi=d)

plt.plot(x, y1,marker='', color='olive', linewidth=2, linestyle='dashed', label="SVM_MC")
plt.plot(x, y2, color='green', linewidth=3,  label="CRF")
plt.legend()

plt.savefig("wordaccuracies.png")


