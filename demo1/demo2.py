'''
editor: Jones
date: 2018/10/09
content: tensorflow demo2
'''

import tensorflow as tf 
# import numpy as np 


maxtrix1 = tf.constant([[3,3]])
maxtrix2 = tf.constant([[2],[2]])

product = tf.matmul(maxtrix1, maxtrix2) # matrix multiply 

# # method1
# sess = tf.Session()
# result1 = sess.run(maxtrix1)
# result2 = sess.run(maxtrix2)
# print result1, result2
# sess.close()

# method2
with tf.Session() as sess:
	result = sess.run(product)
	print result