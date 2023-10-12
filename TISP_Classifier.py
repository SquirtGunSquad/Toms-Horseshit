#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from tabulate import tabulate


# In[2]:


def err(xw,y):
    return metrics.accuracy_score((y>0), (xw>0), normalize=False)/xw.shape[0]
def grad(xw,y):
    pofy=1/(1+np.exp(-xw))
    g=x.T@(y.reshape(-1)-pofy.reshape(-1))
    return g
def preprocess(x,xt,y,yt) #This function processes imported data into a form supported by the TISP classifier.
    #ALL INPUT VARIABLES MUST BE NUMPY ARRAYS
    #IF THEY ARE PANDAS DATAFRAMES UNCOMMENT THE FOLLOWING CODE:
    #x = x.copy().to_numpy()
    #y = y.copy().to_numpy()
    #xt = xt.copy().to_numpy()
    #yt = yt.copy().to_numpy()
    #x and xt are your training and test features, respectively
    #y and yt are your training and test targets, respectively
    sx=np.std(x,axis=0)
    x=x[:,sx>0]
    xt=xt[:,sx>0]
    mx=np.mean(x,axis=0)
    sx=np.std(x,axis=0)
    x=(x-mx)/sx
    xt=(xt-mx)/sx
    y[y<0]=0
    yt[yt<0]=0
    n=x.shape[0]
    nt=xt.shape[0]
    x=np.concatenate((np.ones((n,1)),x),axis=1)
    xt=np.concatenate((np.ones((nt,1)),xt),axis=1)
    return x, xt, y, yt, n, nt


# In[3]:


def TISP_Class(x,xt,y,yt,la,r) : #la variable is decided by the user. It controls the tolerance of the algorithm.
    #larger values of lambda leads to fewer variables being selected for the final model.
    #r controls the number of iterations of the TISP algorithm.
    #It doesn't need to be very high to reach convergence. Larger values will lead to longer CPU time. I recommend about 100.
    p=x.shape[1]
    w=np.zeros((p))
    xw=x@w
    g=grad(xw,y)
    its=[]
    train_error=[]
    test_error=[]
    nfeatures=[]
    l = la
    for it in range(r):
        xw=x@w
        g=grad(xw,y)
        w=w+g/n
        xwt=xt@w
        w[np.abs(w)<l] = 0
        train_error.append(1 - err(xw,y))
        test_error.append(1 - err(xwt,yt))
        its.append(it)
        nfeatures.append(np.shape(w[w!=0]))
        plot1_stuff = [its,train_error]
        plot2_stuff = [its,test_error]
        plot3_stuff = [nfeatures, train_error]
        plot4_stuff = [nfeatures, test_error]
    fig2, ax2 = plt.subplots()
    ax2.plot(plot1_stuff[0],plot1_stuff[1], label = 'train_error')
    ax2.set_title("Misclassification Error vs Number of Iterations")
    ax2.set_xlabel("Number of Iterations")
    ax2.set_ylabel("Misclassification Error")
    ax2.legend()
    fpr, tpr, thresholds = metrics.roc_curve(y, xw)
    fprt, tprt, thresholds = metrics.roc_curve(yt, xwt)
    fig3, ax3 = plt.subplots()
    ax3.plot(fpr,tpr)
    ax3.plot(fprt,tprt)
    ax3.set_title("ROC Curve")
    fig1, ax1 = plt.subplots()
    ax1.plot(plot3_stuff[0], plot3_stuff[1], label = 'train_error')
    ax1.plot(plot4_stuff[0], plot4_stuff[1], label = 'test_error')
    ax1.set_title("Misclassification Error vs Number of Features")
    ax1.set_xlabel("Number of Features")
    ax1.set_ylabel("Misclassification Error")
    ax1.legend()
    table_data = [["Training Error", train_error[-1]], ["Test Error",test_error[-1]], ["Lambda",l], ["Number of Features",nfeatures[-1][0]]]
    print(tabulate(table_data))
    return xw, xwt

