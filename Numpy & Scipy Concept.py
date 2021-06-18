#!/usr/bin/env python
# coding: utf-8

# # 1-dimensional shape

# In[1]:


import numpy as np


# In[3]:


a=np.array([1,2,3])
print(a)


# In[4]:


a.shape


# In[5]:


a.ndim


# In[6]:


len(a)


# # 2-dimensional shape

# In[9]:


b=np.array([[1,2,3],[4,5,6]])
print(b)


# In[10]:


b.shape


# In[11]:


b.ndim


# In[12]:


len(b)


# # 3-dimensional shape

# In[15]:


c=np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(c)


# In[16]:


c.shape


# In[17]:


c.ndim


# In[18]:


len(c)


# # Initialize all the elements of x X y array to 0

# In[41]:


import numpy as np


# In[42]:


np.zeros((3,4))  # inside this we provide size


# # Arrange the numbers between x and y with an interval of z

# In[2]:


import numpy as np


# In[3]:


np.arange(1,10,2)   # here x=1, y=10 and 2 is the interval means z value, so number comes between 1 to 10 only


# In[4]:


np.arange(10,25,5)


# In[46]:


# even number between 10 and 20


# In[47]:


np.arange(10,20,2)


# # Arrange 'z' numbers between x and y

# In[49]:


np.linspace(5,10,10)  # here last 10 means the number we want in between 5 to 10


# In[50]:


#even number between 0 and 10


# In[52]:


np.linspace(0,10,6)


# # Filling SAME number in an array of dimension x X y

# In[54]:


np.full((2,3),6)  # here 2 is no of rows and 6 is no of col and the 6 is the number we want to print in the whole matrix


# In[55]:


np.full((2,5),2)


# # Filling RANDOM number in an array of dimension x X y

# In[58]:


np.random.random((2,2))


# # Inspecting the array: Checking the size of the array

# In[60]:


a=np.array([[2,3,4],[4,4,6]])
a.shape


# In[9]:


import numpy as np
s=np.array([[2,3,4,5],[6,7,8,9],[1,3,5,7]])
print(s.shape)


# # Inspecting the array: Resize the Array

# In[63]:


a=np.array([[2,3,4],[4,4,6]])
a.shape=(3,2)
a


# In[64]:


a = np.array([[2,3,4,4],[2,4,4,6]])
a.shape = (8,1) #Trick: x*y = Total number of elements in the array
a


# # Return the dimension of the array

# In[70]:


a = np.arange(24)
a
#a.ndim
#reshape our array
#b = a.reshape(12,2) #trick: Calculate the factors of 24: 1,2,3,4,6,12,24 
#b.ndim


# In[71]:


a = np.arange(24)
a
a.ndim


# In[75]:


a = np.arange(24)
a
#a.ndim
#reshape our array
b = a.reshape(12,2) #trick: Calculate the factors of 24: 1,2,3,4,6,12,24 
b.ndim
#b


# # Find the number of elements in an array

# In[77]:


a.size


# In[78]:


d = np.array([[1,2,3,4],[4,5,6,4],[6,7,8,9]])   # for getting perfect size for all the elements in the data then data should be equal in range then it will dispaly the size otherwise it will show the total number of rows only. 
d.size


# # Find the datatype of the array

# In[11]:


a=np.arange(24,dtype=float)
print(a.dtype)
a


# In[12]:


a=np.arange(24,dtype=int)
print(a.dtype)
a


# # Numpy Array Mathematics: Addition

# In[1]:


import numpy as np


# In[2]:


np.sum([10,20])


# In[3]:


#Using a variable that is sum a+b
a,b = 10,20
np.sum([a+b])


# In[4]:


np.sum([[1,2],[5,6]], axis=0)


# In[5]:


np.sum([[1,2],[5,6]], axis=1)


# In[6]:


np.sum([[1,2],[5,6]])


# # Numpy Array Mathematics: Subtraction
# 

# In[2]:


import numpy as np
np.subtract(10,20)


# # All other numpy Mathematics Function

# In[3]:


np.multiply(2,3) #Multiply two numbers


# In[4]:


np.divide(6,2) #Divide two numbers


# In[5]:


a=np.array([2,4,6])
b=np.array([1,2,3])
np.multiply(a,b)


# In[6]:


a=np.array([2,4,6])
b=np.array([1,2,3])
np.divide(a,b)


# In[8]:


#exp,sqrt,sin,cos,log
a=np.array([2,4,6])
print("Exponent : ",np.exp(a))
print("Square root : ", np.sqrt(a))
print("Sin : ", np.sin(a))
print("Cos : ", np.cos(a))
print("Log : ", np.log(a))


# # Array Comparison

# In[9]:


#Element-wise Comparison
a = [1,2,4]
b = [2,4,4]
c = [1,2,4]
np.equal(a,b)


# In[10]:


a = [1,2,4]
b = [2,4,4]
c = [1,2,4]
np.equal(a,c)


# In[11]:


#Array-wise Comparison     --> all the elements are true then it will return true if one of the element is false it will return false 
a = [1,2,4]
b = [1,4,4]
c = [1,2,4]
np.array_equal(a,c)


# In[12]:


#Array-wise Comparison     --> all the elements are true then it will return true if one of the element is false it will return false 
a = [1,2,4]
b = [1,4,4]
c = [1,2,4]
np.array_equal(a,b)


# #  Aggregate Function

# In[13]:


a = [1,2,4]
b = [2,4,4]
c = [1,2,4]
print("Sum: ",np.sum(a))
print("Minimum Value: ",np.min(a))
print("Mean: ",np.mean(a))
print("Median: ",np.median(a))
print("Coorelation Coefficient: ",np.corrcoef(a))
print("Standard Deviation: ",np.std(a))


# # Concept of Broadcasting

# In[14]:


#in this case number of rows and number of columns in each array or should be same and one of the should be same value --- broadcasting


# In[15]:


a=np.array([[0,0,0],[1,2,3],[4,5,6],[5,6,7]])
b=np.array([0,1,2])
print("First Array: \n",a,'\n')
print("Second Array: \n",b,'\n')
print("First Array + Second Array: \n",a+b,'\n')


# # Array Manipulation in Python

# In[16]:


#concatenation of two arrays

a = np.array([1,2,3])
b= np.array([4,5,6])
np.concatenate((a,b))


# In[17]:


#Stack array row-wise: Horizontal 

a = np.array([1,2,3])
b= np.array([4,5,6])
np.hstack((a,b))


# In[18]:


#Stack array row-wise: Vertically

a = np.array([1,2,3])
b= np.array([4,5,6])
np.vstack((a,b))


# In[19]:


#Combining Column-wise

a = np.array([1,2,3])
b= np.array([4,5,6])
np.column_stack((a,b))


# # Splitting Array

# In[21]:


x = np.arange(16).reshape(4,4)
print(x,"\n\n")
print(np.hsplit(x,2))
print("\n\n", np.hsplit(x,np.array([2,3])))


# In[23]:


x = np.arange(16).reshape(4,4)
print(x,"\n\n")
print(np.vsplit(x,2))
print("\n\n", np.vsplit(x,np.array([2,3])))


# # Indexing and Slicing in Python 

# In[22]:


# indexing

a = ['m','o','n','t','y',' ','p','y','t','h','o','n']
a[2:9]


# In[23]:


# extract or slicing

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
a[0]
a[:1]
print(a)
a[:1,1:]


# In[24]:


a[:2,1:] 


# In[25]:


a[1:,1:]


# # Advantages of Numpy Over a List

# In[26]:


#Numpy vs List: Memory size
import numpy as np
import sys

#define a list
l = range(1000)
print("Size of a list: ",sys.getsizeof(1)*len(l))

#define a numpy array
a = np.arange(1000)
print("Size of an array: ",a.size*a.itemsize)


# In[13]:


#Numpy vs List: Speed
import time
def using_List():
    t1 = time.time()#Starting/Initial Time
    X = range(10000)
    Y = range(10000)
    z = [X[i]+Y[i] for i in range(len(X))]
    return time.time()-t1 #process time

def using_Numpy():
    t1 = time.time()#Starting/Initial Time
    a = np.arange(10000)
    b = np.arange(10000)
    z =a+b #more convient than a list
    return time.time()-t1
list_time = using_List()
numpy_time = using_Numpy()
print(list_time,numpy_time)
print("In this example Numpy is "+str(list_time/numpy_time)+" times faster than a list")


# In[4]:


import numpy as np
a=np.ones([3,5])
a


# In[5]:


a=np.zeros([2,3])
a


# In[11]:


np.eye(3)


# In[12]:


np.ones(5)


# In[13]:


np.eye(16)


# In[14]:


np.dot(4,5)


# In[15]:


np.vdot([1,2],[3,4])


# In[18]:


a = np.array([[1,2],[3,4]])
b = np.array([[2,4],[7,6]])
np.inner(a,b)


# In[19]:


a = np.array([[1,2],[3,4]])
b = np.array([[2,4],[7,6]])
np.outer(a,b)


# In[22]:


a = np.array([[1,2],[3,4]])
b = np.array([[2,4],[7,6]])
np.matmul(a,b)    # matrix multiplication


# In[ ]:




