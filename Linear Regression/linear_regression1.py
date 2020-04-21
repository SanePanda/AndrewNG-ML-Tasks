import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

plt.rcParams.update({'font.size': 20, 'figure.figsize': (10, 8)})
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
filepath = os.path.join(__location__, 'ex1data1.txt')

df = pd.read_csv(filepath)
mydata = df.to_numpy()
m = mydata.shape[0]

xvals = mydata[:,[0]]
xzero = np.ones((m,1))
xvals = np.hstack((xzero,xvals)) #same as adding two cols

yvals = mydata[:,[1]]

theta = np.zeros((2,1))

def computeCost(x,y,theta):
    m = len(y)
    pred = x.dot(theta)
    sqerr = (pred - y)**2
    return 1/(2*m) * sqerr.sum()

print(computeCost(xvals,yvals,theta)) 

def hypo(th,x):
    return np.matmul(x,theta)

def plot_graph(params):

    plt.xticks(np.arange(5,30,step=5))
    plt.yticks(np.arange(-5,30,step=5))
    xaxis = np.arange(0.0,25,0.1)
    yaxis = xaxis * params[1,0]
    yaxis += params[0,0]
    
    plt.plot(xaxis,yaxis)
    plt.scatter(xvals[:,[1]], yvals)
    plt.show()


alpha = 0.01
cost = []

for i in range(1500):
    cost.append(computeCost(xvals,yvals,theta))
    predictions = xvals.dot(theta)
    error = np.matmul(xvals.transpose(),(predictions -yvals))
    descent=alpha * 1/m * error
    theta-=descent

# print(theta[0,0])
plt.plot(cost)
plt.show()
plot_graph(theta)
print(computeCost(xvals,yvals,theta))