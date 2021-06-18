#!/usr/bin/env python
# coding: utf-8

# # Python-SciPy Library

# IMPORT DATA

# In[5]:


import pandas as pd
#importing data
data = pd.read_excel('C:/Users/AMIT-/OneDrive/Documents/somecars1.xlsx')
#print the dataset
data


# # scipy.cluster

# To divide a dataset into k clusters

# In[6]:


#import libraries
import pandas  as pd
from  scipy.cluster.vq import kmeans, vq
#importing data
data = pd.read_excel('C:/Users/AMIT-/OneDrive/Documents/somecars1.xlsx') 
#find out centroids with the help of kmeans functions
#k, number of clusters required
centroid, _ = kmeans(data,3)
#find out the cluster index for each record with vector quantization function
#vq(data,centroid)
idx, _ = vq(data,centroid)
#print the cluster index array
idx


# In[7]:


#also print the centroids
centroid


# Now perform data whitening

# In[8]:


#import libraries
import pandas  as pd
from  scipy.cluster.vq import kmeans, vq, whiten 
#importing data
data = pd.read_excel('C:/Users/AMIT-/OneDrive/Documents/somecars1.xlsx') 
#whiten data
data = whiten(data)
#find out centroids with the help of kmeans functions
#k, number of clusters required
centroid, _ = kmeans(data,3)
#find out the cluster index for each record with vector quantization function
#vq(data,centroid)
idx, _ = vq(data,centroid)
#print the cluster index array
idx


# In[9]:


#also print the centroids
centroid


# # scipy.stats

# In[8]:


#import numpy
import numpy as np 
#create the marks array 
coffee = np.array([15,18,20,26,32,38,32,24,21,16,13,11,14])
print(coffee.mean(), coffee.std())
#let us see the data distribution by plotting it
import matplotlib.pyplot as plt
plt.plot(range(13),coffee)


# In[7]:


#import numpy
import numpy as np 
#create the marks array 
coffee = np.array([15,18,20,26,32,38,32,24,21,16,13,11,14])
from scipy import stats
#find the zscore
print(stats.zscore(coffee))
print(coffee.mean(), coffee.std())
#let us see the data distribution by plotting it
import matplotlib.pyplot as plt
plt.plot(range(13),coffee)


# In[9]:


#import numpy
import numpy as np 
#create the marks array 
coffee = np.array([15,18,20,26,32,38,32,24,21,16,13,11])
#import scipy stats
from scipy import stats
#find the zscore
print(coffee.mean(), coffee.std())
#let us see the data distribution by plotting it
import matplotlib.pyplot as plt
plt.plot(range(12),coffee)


# In[10]:


#import numpy
import numpy as np
from scipy import stats
#create the numpy array consisting of frequency of people going to gym and frequency of smoking  
obs = np.array([[7,1,3],[87,18,84],[12,3,4],[9,1,7]])
#since we are lookingfor only p values, ignore the rest
_,p,_,_ = stats.chi2_contingency(obs)
#print p
p


# # scipy signal

# In[12]:


#scipy.signal uses FFT to resample a 1D signal.
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
#Now let us create a signal with 200 data point
t = np.linspace(-10, 10, 200) #Defining Time Interval
y = np.sin(t)                         
x_resampled=signal.resample(y, 100) #Number of required samples is 100
plt.plot(t, y)
#for x axis slice t into 2 step size
plt.plot(t[::2], x_resampled, 'o') 
plt.show()


# In[13]:


import numpy as np
t = np.linspace(-10, 10, 200)
x = np.sin(t)
from scipy import signal
x_resampled = signal.resample(x, 25) # Number of required samples is 25
plt.plot(t, x) 
plt.plot(t[::8], x_resampled, 'o')
plt.show()


# # scipy.optimize

# In[14]:


#generate one function and plot with matplotlib
#import matplotlib
import matplotlib.pyplot as plt
#import numpy
import numpy as np
x= np.arange(0.0,1.0,0.1)
#create function
def f(x):
    return -np.exp(-(x-0.7)**2)
#plot function
plt.plot(x,f(x),'o-')
plt.grid()


# In[15]:


#find at which x value we get the minimum function
from scipy import optimize
#generating the function
import numpy as np
def f(x):
    return -np.exp(-(x-0.7)**2)
#find the minimum of the function
result = optimize .minimize_scalar(f)
#now find the corresponding x value
x_min = result.x
#print the x value
x_min


# # scipy.integrate

# In[16]:


#import scipy integrate
import scipy.integrate as intg
#create one function to find the integration
def integrad(x):
    return x**2
#apply quad() function, get only the answer, ignore rest
ans,_ = intg.quad(integrad,0,1)
#print ans
ans


#  # scipy.fftpack

# In[18]:


# create one noisy signal

import matplotlib.pyplot as plt
import numpy as np
#create a signal with time_step=0.02
time_step = 0.02
period = 5
time_vec= np.arange(0,20, time_step)
sig = np.sin(2*np.pi/period*time_vec)+ 0.5*np.random.randn(time_vec.size)
plt.plot(time_vec,sig)
plt.show()


# In[20]:


# Apply fft

from scipy import fftpack
#Since we didnt not know the signal frequency, we only knew the sampling time step of the signal sig.
#The function fftfreq() returns the FFT sample frequency points.
sample_freq = fftpack.fftfreq(sig.size, d = time_step)
#now apply the fft() in the signal to find the discrete fourier transform
sig_fft = fftpack.fft(sig)
#Calculate the absolute value element-wise
power = np.abs(sig_fft)
plt.figure(figsize=(20,5))
#plot the absolute values of each sample_freq
plt.plot(sample_freq, power)
plt.show()
#here at sample_freq = 0.2 and -0.2 we have absolute values of 5.15943859e+02 = 515.9438593147901 
#print(sample_freq)
#print(power)


# In[21]:


# Apply inverse fft

#Filter out the sample frequencies that are greater than 0 with numpy.where(condition) 
pos_mask = np.where(sample_freq > 0)
#Apply the fiter on smaple_freq and store the +ve sample_freq on freqs 
freqs = sample_freq[pos_mask]

#print(power[pos_mask].argmax())
#Find the peak frequency, here we focus on only the positive frequencies
peak_freq = freqs[power[pos_mask].argmax()]
#now get an array copy of the signal where we already applied fft.
high_freq_fft = sig_fft.copy()
#assign the ones greater than peak freq as 0 in order to remove the noise
high_freq_fft[np.abs(sample_freq) > peak_freq] = 0

#print(high_freq_fft)
#Now apply inverese fft on the new high_freq_fft this will be the filtered signal
filtered_sig = fftpack.ifft(high_freq_fft)
#plot 
plt.figure(figsize=(6, 5))
#now plot the original signal for reference
plt.plot(time_vec, sig, label='Original signal')
#now plot the filtered signal
plt.plot(time_vec, filtered_sig, linewidth=3, label='Filtered signal')
#add label, legend
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend(loc='best')
#show
plt.show()


# # scipy.linalg

# In[22]:


# Determinant of a square matrix

#import scipy linalg package
from scipy import linalg
#import numpy to the square matrix
import numpy as np
data = np.array([[1,2,3],[3,4,5],[5,6,7]])
#find determinant 
linalg.det(data)


# In[23]:


# Inverse of a square matrix

#import scipy linalg package
from scipy import linalg
#import numpy to the square matrix
import numpy as np
data = np.array([[1,2,3],[3,4,5],[5,6,7]])
#find determinant 
linalg.inv(data)


# In[24]:


# Eigen values of a square matrix

#import scipy linalg package
from scipy import linalg
#import numpy to the square matrix
import numpy as np
data = np.array([[1,2,3],[3,4,5],[5,6,7]])
#find determinant 
linalg.eigvals(data)


# In[ ]:




