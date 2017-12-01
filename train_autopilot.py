import tensorflow as tf
import numpy as np
import data_set_manager as dsm

from time import sleep, time

# Global variables
tr_size = dsm.HDF5_DataSet.get_tr_set_size()
te_size = dsm.HDF5_DataSet.get_te_set_size()
sample_size = dsm.HDF5_DataSet.get_sample_size()
label_size = dsm.HDF5_DataSet.get_label_size()

epochs = 50
mini_batch_size = 100

# MLP with one hidden layer
input_size = sample_size
hidden_units = 100
output_size = label_size

x = tf.placeholder('float', shape = [None, input_size], name = 'input')
y = tf.placeholder('float', shape = [None, output_size], name = 'output')

w1 = tf.Variable(tf.random_normal([input_size, hidden_units]), name = 'w1')
b1 = tf.Variable(tf.random_normal([hidden_units]), name = 'b1')

w_o = tf.Variable(tf.random_normal([hidden_units, output_size]), name = 'w_o')
b_o = tf.Variable(tf.random_normal([output_size]), name = 'b_o')

def feedforward(x) :

	l1 = tf.add(tf.matmul(x, w1), b1)
	l1 = tf.nn.relu(l1)

	output = tf.add(tf.matmul(l1, w_o), b_o)
	output = tf.nn.sigmoid(output)

	return output

prediction = feedforward(x)
#cost = tf.reduce_mean(tf.pow(tf.subtract(prediction, y), 2))
cost = tf.losses.absolute_difference(y, prediction) 
optimizer = tf.train.AdamOptimizer().minimize(cost)

file_reader = dsm.HDF5_Reader('DataSet.h5')

with tf.Session() as sess :

	sess.run(tf.global_variables_initializer())

	start = time()
	for epoch in range(epochs) :

		epoch_cost = 0
		starting_point = 0

		for batch in range(tr_size / mini_batch_size) :

			sample_batch, label_batch = file_reader.next_batch(starting_point, mini_batch_size, 'train')
			_, c = sess.run( [optimizer, cost], {x: sample_batch, y: label_batch} )

			starting_point += mini_batch_size
			epoch_cost += c

		print 'Epoch ', epoch + 1, ' cost ',  epoch_cost

	print 'finished training in ', time() - start

	# TODO evaluate accuracy

	saver = tf.train.Saver()
	saver.save(sess, 'models/model.ckpt')
	file_reader.close_file()
