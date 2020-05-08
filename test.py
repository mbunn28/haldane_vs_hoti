import scipy.optimize as opt
import joblib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

xv = joblib.load("xvals")
v = joblib.load("values")
a = 0.5/np.sqrt(3)
b=1/np.sqrt(3)
fig = plt.figure()
plt.plot(xv,v)
plt.plot([a,b,0.5],[2,2,2],'rx')
fig.suptitle("Phase Diagram")
plt.xlabel('Lambda')
plt.ylabel('Alpha')
plt.ylim(0,2)
plt.xlim(0,2)
plt.xticks(np.arange(0,2,0.25),('0','0.25','0.5','0.75','1','1/0.75','1/0.5','1/0.25','Inf'))
plt.yticks(np.arange(0,2,0.25),('0','0.25','0.5','0.75','1','1/0.75','1/0.5','1/0.25','Inf'))
fig.savefig(f"output/phase_diagram.pdf")
plt.close(fig)