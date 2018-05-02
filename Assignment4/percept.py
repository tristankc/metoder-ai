from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import csv
import timeit
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def logistic_z(z):
    return 1.0/(1.0+np.exp(-z))

def logistic_wx(w,x):
    return logistic_z(np.inner(w,x))

def classify(w,x):
    x = np.hstack(([1],x))
    return 0 if (logistic_wx(w,x)<0.5) else 1


def stochast_train_w(x_train,y_train,learn_rate=0.1,niter=100):
    x_train = np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim = x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]

    for it in range(niter):
        if(len(index_lst)==0):
            index_lst=random.sample(range(num_n), k=num_n)
        xy_index = index_lst.pop()
        x=x_train[xy_index]
        y=y_train[xy_index]

        for i in range(dim):
            update_grad = (logistic_wx(w,x)-y) * x[i]* np.exp(-np.inner(x,w)) * logistic_wx(w,x)**2
            w[i] = w[i] - learn_rate*update_grad

    return w

def batch_train_w(x_train,y_train,learn_rate=0.1,niter=100):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    for it in range(niter):
        for i in range(dim):
            update_grad=0.0
            for n in range(num_n):
                update_grad += (logistic_wx(w,x_train[n])-y_train[n])* x_train[n][i] * np.exp(-np.inner(w, x_train[n])) * logistic_wx(w, x_train[n])**2
            w[i] = w[i] - learn_rate * update_grad/num_n
    return w

def train_and_plot(xtrain,ytrain,xtest,ytest,training_method,learn_rate=0.1,niter=10):
    plt.figure()
    #train data
    data = pd.DataFrame(np.hstack((xtrain,ytrain.reshape(xtrain.shape[0],1))),columns=['x','y','lab'])
    ax=data.plot(kind='scatter',x='x',y='y',c='lab',cmap=cm.copper,edgecolors='black')

    #train weights
    start = timeit.default_timer()
    w=training_method(xtrain,ytrain,learn_rate,niter)
    stop = timeit.default_timer()

    error=[]
    y_est=[]
    for i in range(len(ytest)):
        error.append(np.abs(classify(w,xtest[i])-ytest[i]))
        y_est.append(classify(w,xtest[i]))
    y_est=np.array(y_est)
    data_test = pd.DataFrame(np.hstack((xtest,y_est.reshape(xtest.shape[0],1))),columns=['x','y','lab'])
    data_test.plot(kind='scatter',x='x',y='y',c='lab',ax=ax,cmap=cm.coolwarm,edgecolors='black')

    plt.show()

    return w, np.mean(error), stop-start


def read_file_to_list(file):

    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter = '\t')
        csv_list = list(reader)

    x_values = []
    y_values = []

    for x in range(len(csv_list)):
        y_values.append(int(eval(csv_list[x][-1])))
        csv_list[x].pop(-1)
        x_values.append(csv_list[x])

    for x in range(len(x_values)):
        for y in range(len(x_values[x])):
            x_values[x][y] = float(x_values[x][y])

    return np.array(x_values), np.array(y_values)


x_values, y_values = read_file_to_list("data/data_big_separable_train.csv")
x_test, y_test = read_file_to_list("data/data_big_separable_test.csv")



w = train_and_plot(x_values, y_values, x_test, y_test, stochast_train_w, niter=500)
#w = batch_train_w(x_values, y_values, niter=100)
#print("Error: "+ str(w[1]))


average = 0
mean_error = 0
""""
for x in range(1):
    current = train_and_plot(x_values, y_values, x_test, y_test, batch_train_w, niter=100)
    #current = stochast_train_w(x_values, y_values, niter=100)
    #current = batch_train_w(x_values, y_values, niter=100)
    mean_error += current[1]
    average += current[2]

print(mean_error/1)
print(average/1)
"""