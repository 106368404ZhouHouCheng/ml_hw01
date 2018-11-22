'''
editor: Jones
date: 2018/10/09
content: tensorflow demo4
'''

import tensorflow as tf 

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
	print sess.run(output, feed_dict = {input1:[6.], input2:[2.]})


