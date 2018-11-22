'''
editor: Jones
date: 2018/10/09
content: tensorflow demo10 THE MNIST DATABASE of handwritten digits
'''

import numpy as np 
import pandas as pd
import keras
# from kears.utils import np_utils
from keras.utils import to_categorical

(x_Train, y_Train), (x_Test, y_Test) = keras.datasets.mnist.load_data()

print 'x_train_image:', x_Train.shape
print 'y_Train_label:', y_Train.shape

print('x_test_image:',x_Test.shape)
print('y_test_label:',y_Test.shape)
