'''
editor: Jones
date: 2018/10/09
content: tensorflow demo6
'''

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 


def add_layer(inputs, in_size, out_size, activation_function = None):

	with tf.name_scope('layer'):
		with tf.name_scope('Weights'):
			Weights = tf.Variable(tf.random_normal([in_size, out_size]), name = 'W')
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name = 'b')
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

		if activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)
		return outputs


# x_data = np.linspace(-1, 1, 300)[:, np.newaxis].astype(np.float32)
# noise = np.random.normal(0, 0.05, x_data.shape)
# y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network 
with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32, [None, 1], name = 'x_input')
	ys = tf.placeholder(tf.float32, [None, 1], name = 'y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)
# add output layer
predition = add_layer(l1, 10, 1, activation_function = None)

# the error between prediction and real data
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition), reduction_indices = [1]), name = 'loss')
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# # init = tf.initialize_all_variables()
# if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
#     init = tf.initialize_all_variables()
# else:
#     init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter("log/", sess.graph)
# important step
sess.run(tf.initialize_all_variables())

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(x_data, y_data)
# plt.ion()
# plt.show()

# for i in range(1000):
# 	sess.run(train_step, feed_dict = {xs: x_data, ys: y_data})
# 	if i % 50 == 0:
# 		# print sess.run(loss, feed_dict = {xs: x_data, ys: y_data})
# 		try:
# 			ax.lines.remove(lines[0])
# 		except Exception as e:
# 			pass
# 		predition_value = sess.run(predition, feed_dict = {xs: x_data})
# 		lines = ax.plot(x_data, predition_value, 'r-', lw = 5)
# 		ax.lines.remove(lines[0])
# 		plt.pause(0.1)












