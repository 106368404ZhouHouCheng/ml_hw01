'''
editor: Jones
date: 2018/10/11
content: read train_fit.csv file
'''

# -*- coding: utf-8 -*-

import csv
from random import shuffle
from tqdm import tqdm


# define a function of open csv.file and str to int type 



# # label:0 top
# #      :1 back
# #      :2 right
# #      :3 left
# #      :4 test
# def label_posture(item):
#     if item == 0:
#         return [1,0,0,0,0]
#     elif item == 1:
#         return [0,1,0,0,0]
#     elif item == 2:
#         return [0,0,1,0,0]
#     elif item == 3:
#         return [0,0,0,1,0]
#     else:
#         return [0,0,0,0,1]

# # create train data
# def create_train_data():

#     train_set, test_set = diff_dataset()
#     # print test_set[0]
#     training_data = []
#     n = 8 # 128 divided into 16 columns

#     for i in tqdm(train_set):
#         label = label_posture(i[0])
#         row_list = [i[j:j + n] for j in range(0, len(i[1:]), n)]
#         training_data.append([np.array(row_list), np.array(label)])

#     shuffle(training_data)
#     np.save('train_data.npy', training_data)
#     return training_data


# # create test data
# def create_test_data():

#     train_set, test_set = diff_dataset()
#     testing_data = []
#     n = 8 # 128 divided into 16 columns
#     num = 0

#     for i in tqdm(test_set):       
#         row_list = [i[j:j + n] for j in range(0, len(i), n)]
#         testing_data.append([np.array(row_list), np.array(num)])
#         num += 1

#     np.save('test_data.npy', testing_data)
#     return testing_data


# def diff_dataset():

#     str_to_int_list = str_to_int()
#     test_list = []
#     train_list = []

#     for i in str_to_int_list:
#         for j in i[:1]:
#             if j == 4:
#                 test_list.append(i[1:])
#             else:
#                 train_list.append(i)
#     return train_list, test_list


# # str to int and remove the first item
# def str_to_int():

#     csv_reader_list = open_file(dir_path)
#     str_to_int_list = []
    
#     for i in csv_reader_list[1:]:
#         row_list = []
#         for j in i:
#             row_list.append(int(j))

#         str_to_int_list.append(row_list)

#     return str_to_int_list


# # open the csv file
# def open_file(path):

#     with open(path) as csv_file:
#         csv_reader = csv.reader(csv_file)
#         csv_reader_list = []
#         for row in csv_reader:
#             csv_reader_list.append(row)

#     return csv_reader_list


# dir_path = 'train_fix.csv'
# train_set = create_train_data()
# test_set = create_test_data()

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import *
from sklearn.preprocessing import *
from sklearn.utils import shuffle

# load data
csv_data = pd.read_csv('train_fix.csv')
# split data as train data, test data and vaild data
row_train_data = csv_data[:2207]
train_data_shu = shuffle(row_train_data)

train_data = train_data_shu[:-100]
valid_data = train_data_shu[2107:]

X_train = train_data.drop(['label'], axis=1).values
Y_train = train_data['label'].values

X_valid = valid_data.drop(['label'], axis=1).values
Y_valid = valid_data['label'].values

test_data = csv_data[2207:]

X_test = test_data.drop(['label'], axis=1).values

# normalized
X_train = scale(X_train)
X_valid = scale(X_valid)
X_test = scale(X_test)

print X_train.shape
print X_valid.shape
print X_test

# building model
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1],  kernel_initializer='normal',activation='relu'))
model.add(Dense(128, input_dim=32,  kernel_initializer='normal',activation='relu'))
model.add(Dense(128, input_dim=128,  kernel_initializer='normal',activation='relu'))
model.add(Dense(128, input_dim=128,  kernel_initializer='normal',activation='relu'))
model.add(Dense(32, input_dim=128,  kernel_initializer='normal',activation='relu'))
model.add(Dense(X_train.shape[1], input_dim=32,  kernel_initializer='normal',activation='relu'))
model.add(Dense(1,  kernel_initializer='normal'))

model.compile(loss='MAE', optimizer='adam')

nb_epoch = 50
batch_size = 32

save_str_1=str(batch_size)+"_"+str(nb_epoch)
TB=TensorBoard(log_dir='logs/'+save_str_1, histogram_freq=0)
# ready train
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1,validation_data=(X_valid, Y_valid))
# save csv
model.save('h5/'+save_str_1+'.csv')
# prediction
Y_predict = model.predict(X_test)
np.savetxt('test1026.csv', Y_predict, delimiter = ',')
























