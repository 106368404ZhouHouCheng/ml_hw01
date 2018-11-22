'''
editor: Jones
date: 2018/10/03
content: use opencv read image and resize the image size 
'''

import cv2 
import numpy as np 

img = cv2.imread('0.jpg')

height = np.size(img, 0)
width = np.size(img, 1)

# print height
# print width
res = cv2.resize(img,(width/16, height/16), interpolation = cv2.INTER_CUBIC)

height = np.size(res, 0)
width = np.size(res, 1)

print height
print width


cv2.imshow('img',res)
cv2.waitKey(0)
cv2.destroyAllWindows()



