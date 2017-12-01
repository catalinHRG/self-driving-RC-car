import tensorflow as tf
import socket as sk
import numpy as np
import connectivity_manager as cm
import data_set_manager as dsm

from time import time, sleep
from sys import getsizeof

# laying out TF model

input_layer_size = dsm.HDF5_DataSet.get_sample_size()
hidden_layer_size = 150
output_layer_size = dsm.HDF5_DataSet.get_label_size()

x = tf.placeholder('float', [None, input_layer_size])

w1 = tf.Variable(

	tf.zeros( [input_layer_size, hidden_layer_size] ),
	dtype = tf.float32,
	name = 'w1'

	)

b1 = tf.Variable(

	tf.zeros( [hidden_layer_size] ),
	dtype = tf.float32,
	name = 'b1'

	)

w_o = tf.Variable(

	tf.zeros( [hidden_layer_size, output_layer_size] ),
	dtype = tf.float32,
	name = 'w_o'

	)

b_o = tf.Variable(

	tf.zeros( [output_layer_size] ),
	dtype = tf.float32,
	name = 'b_o'

	)

def autopilot(x):

	first_layer = tf.add(tf.matmul(x, w1), b1)
	first_layer = tf.nn.relu(first_layer)

	output_layer = tf.add(tf.matmul(first_layer, w_o), w_o)
	output_layer = tf.nn.softmax( output_layer )

	max_index = tf.argmax(output_layer)
	prediction = np.zeros( len( output_layer ), dtype = np.uint8 )
	np.put(prediction, max_index, 1)

	return prediction # one-hot encoded vector

compute_prediction = autopilot(x)

# loading TF model from disk

with tf.Session as sess :

	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	saver.restore(sess, "models/model.ckpt")

client = cm.connct_to_server("192.168.0.102", 5000)

total = 38421
chunk_size = 4096

steering_vector = np.asarray( [0, 0, 0], dtype = np.uint8 )
raw_data = steering_vector.tostring()

while True :

	print 'Sending : ', steering_vector, ' size : ', getsizeof(raw_data)
	client.sendall( raw_data )

	raw_data = cm.collect_bytes(client, total, chunk_size)

	frame = np.fromstring( raw_data, dtype = np.uint8 )
	frame = frame.astype( dtype = np.float32 )
	frame = np.asarray( [ frame ] )

	with tf.Session() as sess :

		prediction = sess.run( [ compute_prediction ], feed_dict = { x: frame } )

	steering_vector = dsm.convert_to_steering_vector( prediction )
	raw_data = steering_vector.tostring()

