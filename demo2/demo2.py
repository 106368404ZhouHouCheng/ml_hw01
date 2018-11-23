'''
editor: Jones
date: 2018/10/12
content: use numpy, pandas, matplot read 2-D list
'''

#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np 
import pandas as pd 
import scipy as sp
from scipy.optimize import leastsq
import cv2
import matplotlib.pyplot as plt


row_data = pd.read_csv('train_fix.csv').values
# print type(row_data)
data = row_data[30]
# print data
data = data[0:]
# print type(data)

data = np.delete(data,[0],None).reshape([16,8])
print data

print np.where(data == np.max(data))

imgplot1 = plt.imshow(data)
plt.show()

# list1 = []

# for item in data:
# 	j = np.where(item == np.max(item))[0].tolist()
# 	list1.append(j)

# Yi = np.array(list1).T[0]
# Xi = np.arange(len(Yi))

# print "Xi:",Xi,"\nYi:",Yi

# use the least square estimation algorithm to recognition the lateral position
# def func(p, x):
# 	a,b,c = p
# 	return a*x**2 + b*x + c

# def error(p, x, y, s):
# 	print s
# 	return func(p, x) - y

# # def func(p, x):
# # 	a,b = p
# # 	return a*x + b

# # def error(p, x, y, s):
# # 	print s
# # 	return func(p, x) - y


# p0 = [5,2,10]

# s = "Test the number of iteration"


# Para=leastsq(error,p0,args=(Xi,Yi,s))

# a,b,c = Para[0]
# print "a=",a,'\n',"b=",b,"c=",c

# ### matplot, the result of fitting ###

# plt.figure(figsize=(8, 6))
# plt.scatter(Xi, Yi, color = 'red', label = 'Sample Point', linewidth = 1)
# x=np.linspace(0,16,1000)
# y = a*x**2 + b*x + c

# plt.plot(x,y,color="orange",label="Fitting Curve",linewidth=2)
# plt.legend()
# plt.show()































# cv2.imwrite('hung_andy.png', data)

# pic = cv2.imread('hung_andy.png')
# pic = cv2.resize(pic, (1024, 2048), interpolation=cv2.INTER_AREA)
# cv2.imwrite('andy_hung.png', pic)