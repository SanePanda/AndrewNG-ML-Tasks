import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
filepath = os.path.join(__location__, 'ex1data2.txt')


df = pd.read_csv(filepath, header = None)
data_set = df.to_numpy()
m = data_set.shape[0]
n = data_set.shape[1]

xvals = data_set[:,0:n-1:1]
xzero = np.ones((m,1))
xvals = np.hstack((xzero, xvals))

yvals = data_set[:,[-1]]

params = np.zeros((n,1))

def normalise(xvals, yvals):
    n = xvals.shape[1]
    for i in range(1,n):
        xvals[:,i] = (xvals[:,i] - xvals[:,i].mean())/(xvals[:,i].max() - xvals[:,i].min())
    yvals = (yvals - yvals.mean())/(yvals.max() - yvals.min())
    return xvals, yvals

def cost_function(xvals, yvals, params):
    m = xvals.shape[0]
    cost = xvals.dot(params) - yvals
    cost = cost ** 2
    total_cost = cost.sum() / (2 * m)
    return total_cost

def gradients(xvals, yvals, params):
    grad = (xvals.dot(params) - yvals)
    grad = xvals.transpose().dot(grad)
    return grad

def predictions(xvals):
    return xvals.dot(params)

def run_model(xvals, yvals, params, iterations, alpha):
    cost_variation = []
    m = xvals.shape[0]
    k = iterations / 10
    xvals, yvals = normalise(xvals,yvals)
    
    for i in range(iterations+1):
        params -= alpha * gradients(xvals,yvals,params) / m
        current_cost = cost_function(xvals,yvals,params)
        cost_variation.append(current_cost)
        if i % k == 0:
            print("Current cost after %i iterations is : %f" %(i,current_cost))
    
    plt.plot(cost_variation)
    plt.show()

    return params

params = run_model(xvals, yvals, params, 15000, 0.01)