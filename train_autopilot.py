import tensorflow as tf
import numpy as np
import sys

from utils import data_set_manager as dsm
from utils import model as md

from time import time

tr_size = dsm.HDF5_DataSet.get_tr_set_size()
te_size = dsm.HDF5_DataSet.get_te_set_size()
sample_size = dsm.HDF5_DataSet.get_sample_size()
label_size = dsm.HDF5_DataSet.get_label_size()

epochs = int( sys.argv[1] )
mini_batch_size = int( sys.argv[2] )

file_reader = dsm.HDF5_Reader('utils/DataSet.h5')

tr_batches = tr_size / mini_batch_size
te_batches = te_size / mini_batch_size

input_size = sample_size
h1_size = 150
output_size = label_size

x = tf.placeholder( dtype = 'float' , shape = [None, input_size] , name = 'input')
y = tf.placeholder( dtype = 'float' , shape = [None, output_size] , name = 'output')

l1 = md.fully_connected_layer(x, input_size, h1_size, 'l1')
logits = md.fully_connected_layer(l1, h1_size, output_size, 'logits', tf.identity)

cost = md.build_cost_node(logits, y, 'cost')

optimizer = tf.train.AdamOptimizer( name = 'optimizer' ).minimize(cost)

accuracy = md.build_accuracy_node(logits, y, 'accuracy')

merged_summary_ops = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('tensorboard_data/')

with tf.Session() as sess :

	sess.run( tf.global_variables_initializer() )

	for epoch in range(epochs) :

		start = time()

		starting_point = 0

		for batch in range( tr_batches ) :

			sample_batch, label_batch = file_reader.next_batch(starting_point, mini_batch_size, 'train')
			summary, _, _ = sess.run( [merged_summary_ops, optimizer, cost], {x: sample_batch, y: label_batch} )

			summary_writer.add_summary(summary, batch)

			starting_point += mini_batch_size

		print 'Epoch ', epoch + 1, ' finished in ', time() - start

		starting_point = 0

		for batch in range( tr_batches ) :

			sample_batch, label_batch = file_reader.next_batch(starting_point, mini_batch_size, 'train')
			summary, acc = sess.run( [merged_summary_ops, accuracy], feed_dict = { x: sample_batch, y: label_batch } )

			if( batch < te_batches ) :

				sample_batch, label_batch = file_reader.next_batch(starting_point, mini_batch_size, 'test')
				summary, acc = sess.run( [merged_summary_ops, accuracy], feed_dict = { x: sample_batch, y: label_batch } )

			starting_point += mini_batch_size

	summary_writer.close()

	saver = tf.train.Saver()
	saver.save(sess, 'models/model.ckpt')
	file_reader.close_file()
