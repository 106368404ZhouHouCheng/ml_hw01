'''
edtior:Jones
date: 2018/10/02
this is a story of cat vs dog
'''


# preprocessing

import cv2 as cv
import numpy as np 
import os
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt


TRAIN_DIR = 'train'
TEST_DIR = 'test3'

IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'sleep_posture-{}--{}.model'.format(LR, '2conv-basic')

def label_img(img):
	word_label = img.split('.')[-3]

	if word_label == 'face_up':
		return [1,0,0]
	elif word_label == 'sleep_to_left':
		return [0,1,0]
	elif word_label == 'sleep_to_right':
		return [0,0,1]

	# if word_label == 'sleep_to_left':
	# 	return [1,0]
	# elif word_label == 'sleep_to_right':
	# 	return [0,1]

def create_train_data():
	training_data = []
	for img in tqdm(os.listdir(TRAIN_DIR)):
		label = label_img(img)
		path = os.path.join(TRAIN_DIR, img)
		img = cv.imread(path, cv.IMREAD_GRAYSCALE)
		img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
		training_data.append([np.array(img), np.array(label)])

	shuffle(training_data)
	np.save('train_data.npy', training_data)
	return training_data


def process_test_data():
	testing_data = []
	for img in tqdm(os.listdir(TEST_DIR)):
		path = os.path.join(TEST_DIR, img)
		img_num = img.split('.')[0]
		img = cv.imread(path, cv.IMREAD_GRAYSCALE)
		img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
		testing_data.append([np.array(img), img_num])

	shuffle(testing_data)
	np.save('test_data.npy', testing_data)
	return testing_data


train_data = create_train_data()
test_data = process_test_data()

# If you have already created the dataset:
# train_data = np.load('train_data.npy')
# test_data = np.load('test_data.npy')

# print train_data[0]


# Convolutional Neural Network

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


# convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name = 'input')
# convnet = conv_2d(convnet, 32, 5, activation = 'relu')

# convnet = conv_2d(convnet, 64, 5, activation = 'relu')
# convnet = max_pool_2d(convnet, 5)

# convnet = fully_connected(convnet, 1024, activation = 'relu')
# convnet = dropout(convnet, 0.8)

# convnet = fully_connected(convnet, 2, activation = 'softmax')
# convnet = regression(convnet, optimizer = 'adam', learning_rate = LR, loss = 'categorical_crossentropy', name = 'targets')

# model = tflearn.DNN(convnet, tensorboard_dir = 'log')


# if os.path.exists('{}.meta'.format(MODEL_NAME)):
# 	model.load(MODEL_NAME)
# 	print('model loaded!')

# train = train_data[:-500]
# # print train[0]
# test = train_data[-500:]


# X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# Y = [i[1] for i in train]

# test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# test_y = [i[1] for i in test]


# model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
#     snapshot_step=500, show_metric=True, run_id=MODEL_NAME)


import tensorflow as tf
tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')



if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

train = train_data[:-100]
test = train_data[-100:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]


test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]


model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)


# Visually inspecting our network against unlabeled data


# if you need to create the data:
# test_data = process_test_data()
# if you already have some saved:
# test_data = np.load('test_data.npy')


fig = plt.figure()

for  num, data in enumerate(test_data[:25]):
	# sleep_to_left: [1,0]
	# sleep_to_right: [0,1]

	img_num = data[1]
	img_data = data[0]

	y = fig.add_subplot(5, 5, num+1)
	orig = img_data
	data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
	# print data[0]
	model_out = model.predict([data])[0]
	print model_out

	# if np.argmax(model_out) == 1:
	# 	str_label = 'sleep_to_left'
	# else:
	# 	str_label = 'sleep_to_right'

	if np.argmax(model_out) == 1:
		str_label = 'face_up'
	elif np.argmax(model_out) == 1:
		str_label = 'sleep_to_left'
	else:
		str_label = 'sleep_to_right'


	y.imshow(orig, cmap = 'gray')
	plt.title(str_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)

# plt.show()

with open('sampleSubmission.csv', 'w') as f:
	f.write('id,label\n')

with open('sampleSubmission.csv', 'a') as f:
	for data in tqdm(test_data):
		img_num = data[1]
		img_data = data[0]
		orig = img_data
		data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
		model_out = model.predict([data])[0]
		f.write('{},{}\n'.format(img_num,model_out[1]))




 
