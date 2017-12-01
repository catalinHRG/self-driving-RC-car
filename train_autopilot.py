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

tr_batches = tr_size / 100
te_batches = te_size / 100

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

	return output

prediction = feedforward(x)

cost = tf.nn.softmax_cross_entropy_with_logits(lables = y, logits = prediction, name = 'cost')
optimizer = tf.train.AdamOptimizer().minimize(cost)

accuracy = tf.reduce_mean( tf.cast( tf.equal( tf.argmax(prediction), tf.argmax(y) ), 'int'), name = 'accuracy' )

file_reader = dsm.HDF5_Reader('DataSet.h5')

tr_accuracy = te_accuracy = 0

with tf.Session() as sess :

	sess.run(tf.global_variables_initializer())

	start = time()
	for epoch in range(epochs) :

		epoch_cost = 0
		starting_point = 0

		for batch in range(tr_batches) :

			sample_batch, label_batch = file_reader.next_batch(starting_point, mini_batch_size, 'train')
			_, c = sess.run( [optimizer, cost], {x: sample_batch, y: label_batch} )

			starting_point += mini_batch_size
			epoch_cost += c

		print 'Epoch ', epoch + 1, ' cost ',  epoch_cost
		print 'Finished training in ', time() - start

		starting_point = 0

		for batch in range( tr_batches ) :

			sample_batch, label_batch = file_reader.next_batch(starting_point, mini_batch_size, 'train')
			tr_accuracy = tr_accuracy + accuracy.eval( { x: sample_batch, y: label_batch } )

			if( batch < te_batches ) :

				sample_batch, label_batch = file_reader.next_batch(starting_point, mini_batch_size, 'test')
				te_accuracy = te_accuracy + accuracy.eval( { x: sample_batch, y: label_bitch } )

		tr_accuracy = tr_accuracy / tr_batches
		te_accuracy = te_accuracy / te_batches

		print 'Training accuracy : ', tr_accuracy, ' Test accuracy :  ', te_accuracy

	saver = tf.train.Saver()
	saver.save(sess, 'models/model.ckpt')
	file_reader.close_file()
