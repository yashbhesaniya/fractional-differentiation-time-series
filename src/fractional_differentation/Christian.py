#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


def getWeights(d,size):
    '''
    d:fraction
    k:the number of samples
    w:weight assigned to each samples
    
    '''
    # thres>0 drops insignificant weights
    w=[1.]
    for k in range(1,size):
        w_ = -w[-1]/k*(d-k+1)
        w.append(w_)
    w=np.array(w[::-1]).reshape(-1,1) #sort and reshape the w
    return w


# In[4]:


def weight_by_d(dRange=[0,1], nPlots=11, size=6):
    '''
    dRange: the range of d
    nPlots: the number of d we want to check
    size: the data points used as an example
    w: collection of w by different d value
    '''
    
    w=pd.DataFrame()
    
    for d in np.linspace(dRange[0],dRange[1],nPlots):
        w_=getWeights(d,size=size)
        w_=pd.DataFrame(w_,index=range(w_.shape[0])        [::-1],columns=[d])
        w=w.join(w_,how='outer')
        
    return w
weight_by_d = weight_by_d()
weight_by_d


# In[ ]:




