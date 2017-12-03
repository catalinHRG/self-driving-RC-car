import tensorflow as tf
import socket as sk
import numpy as np
import sys

from utils import connectivity_manager as cm
from utils import data_set_manager as dsm
from utils import model as md
from time import time, sleep

client_ip_address = sys.argv[1]
port = int( sys.argv[2] )

input_layer_size = dsm.HDF5_DataSet.get_sample_size()
hidden_layer_size = 150
output_layer_size = dsm.HDF5_DataSet.get_label_size()

# laying out TF model

x = tf.placeholder('float', [None, input_layer_size])

l1 = md.fully_connected_layer(x, input_layer_size, hidden_layer_size, 'l1')
prediction = md.fully_connected_layer(l1, hidden_layer_size, output_layer_size, 'logits', act = tf.identity)

# loading TF model from disk

with tf.Session() as sess :

	sess.run( tf.global_variables_initializer() )
	saver = tf.train.Saver()
	saver.restore(sess, "models/model.ckpt")

client = cm.connect_to_server(client_ip_address, port)

total = 38421
chunk_size = 4096

steering_vector = np.asarray( [0, 0, 0], dtype = np.uint8 )
raw_data = steering_vector.tostring()

while True :

	print 'Sending : ', steering_vector, ' size : ', sys.getsizeof( raw_data )
	client.sendall( raw_data )

	raw_data = cm.collect_bytes(client, total, chunk_size)

	frame = np.fromstring( raw_data, dtype = np.uint8 )
	frame = frame.astype( dtype = np.float32 )
	frame = np.asarray( [ frame ] )

	with tf.Session() as sess :

		prediction = sess.run( [ compute_prediction ], feed_dict = { x: frame } )

	steering_vector = dsm.convert_to_steering_vector( prediction )
	raw_data = steering_vector.tostring()

