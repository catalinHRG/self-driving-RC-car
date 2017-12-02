import h5py as h5
import numpy as np
import os
import sys

# Super Class
class HDF5_DataSet() :

	_tr_set_size = 5000
	_te_set_size = 1000
	_sample_size = 38400
	_label_size = 7
	_file = None

	def __init__ (self, file_name, mode) :

		if not os.path.exists(file_name) :

			print 'File does not exist , will create it and all the data sets inside !! '
			self._file = h5.File(file_name, 'w')

			self._file.create_dataset('tr_samples', (self._tr_set_size, self._sample_size), 'u1')
			self._file.create_dataset('tr_labels', (self._tr_set_size, self._label_size), 'u1')
			self._file.create_dataset('te_samples', (self._te_set_size, self._sample_size), 'u1')
			self._file.create_dataset('te_labels', (self._te_set_size, self._label_size), 'u1')

			self._file.close()

			print 'Closing file !! '

		print 'Opening file for ', ( 'read only' if mode == 'r' else 'append only' ) 
		self._file = h5.File(file_name, mode)

		print 'Getting data set handlers ... '
		self.tr_samples = self._file['/tr_samples']
		self.tr_labels = self._file['/tr_labels']
		self.te_samples = self._file['/te_samples']
		self.te_labels = self._file['/te_labels']
		print 'Got all data set handlers'

	#@classmethod
	def close_file(self) :

		self._file.close()
	
	@classmethod
	def get_tr_set_size(self):

		return self._tr_set_size
	
	@classmethod
	def get_te_set_size(self):

		return self._te_set_size
	
	@classmethod
	def get_sample_size(self):

		return self._sample_size
	
	@classmethod
	def get_label_size(self):

		return self._label_size

class HDF5_Reader(HDF5_DataSet) :

	def __init__ (self, file_name) :

		HDF5_DataSet.__init__(self, file_name, 'r')

	def next_batch(self, starting_point, mini_batch_size, dset):

		if dset == 'train' : 

			samples = self.tr_samples
			labels = self.tr_labels

		elif dset == 'test' :

			samples = self.te_samples
			labels = self.te_labels

		sample_batch = np.empty( [mini_batch_size, self._sample_size] )
		label_batch = np.empty( [mini_batch_size, self._label_size] )

		end_point = starting_point + mini_batch_size
		index_pointer = 0

		for i in range (starting_point, end_point) :

			sample_batch[index_pointer] = samples[i].astype(np.float32)
			label_batch[index_pointer] = labels[i].astype(np.float32)
			index_pointer += 1

		return sample_batch, label_batch

class HDF5_Writer(HDF5_DataSet) :

	def __init__ (self, file_name) :

		HDF5_DataSet.__init__(self, file_name, 'a')

	def insert_entry(self, sample, label, index, dset) :

		if dset == 'train' : 

			self.tr_samples[index] = sample
			self.tr_labels[index] = label

		elif dset == 'test' : 

			self.te_samples[index] = sample
			self.te_labels[index] = label

		else : print 'Invalid'

################ Module wide variables !?

slow_forward_label = np.asarray( [0, 1, 0, 0, 0, 0, 0], dtype = np.uint8 )
slow_left_forward_label = np.asarray( [0, 0, 1, 0, 0, 0, 0], dtype = np.uint8 )
slow_right_left_label = np.asarray( [0, 0, 0, 1, 0, 0, 0], dtype = np.uint8 )

fast_forward_label = np.asarray( [0, 0, 0, 0, 1, 0, 0], dtype = np.uint8 )
fast_left_forward_label = np.asarray( [0, 0, 0, 0, 0, 1, 0], dtype = np.uint8 )
fast_right_forward_label = np.asarray( [0, 0, 0, 0, 0, 0, 1], dtype = np.uint8 )

#######

slow_forward = np.asarray( [0, 0, 50], dtype = np.uint8 )
slow_left_forward = np.asarray( [1, 0, 50], dtype = np.uint8 )
slow_right_forward = np.asarray( [0, 1, 50], dtype = np.uint8 )

fast_forward = np.asarray( [0, 0, 100], dtype = np.uint8 )
fast_left_forward = np.asarray( [1, 0, 100], dtype = np.uint8 )
fast_right_forward = np.asarray( [0, 1, 100], dtype = np.uint8 )

################ Module utility functions

def convert_to_label( steering_vector ) :

	if np.array_equal( steering_vector, slow_forward ) :
		return np.asarray( [0, 1, 0, 0, 0, 0, 0], dtype = np.uint8 )
	elif np.array_equal( steering_vector, slow_left_forward ) :
		return np.asarray( [0, 0, 1, 0, 0, 0, 0], dtype = np.uint8 )
	elif np.array_equal( steering_vector, slow_right_forward ) :
		return np.asarray( [0, 0, 0, 1, 0, 0, 0], dtype = np.uint8 )
	elif np.array_equal( steering_vector, fast_forward ) :
		return np.asarray( [0, 0, 0, 0, 1, 0, 0], dtype = np.uint8 )
	elif np.array_equal( steering_vector, fast_left_forward ) :
		return np.asarray( [0, 0, 0, 0, 0, 1, 0], dtype = np.uint8 )
	elif np.array_equal( steering_vector, fast_right_forward ) :
		return np.asarray( [0, 0, 0, 0, 0, 0, 1], dtype = np.uint8 )
	else :
		return np.asarray( [1, 0, 0, 0, 0, 0, 0], dtype = np.uint8 )

def convert_to_steering_vector( label ) :

	if np.array_equal( label, slow_forward_label ) :
		return np.asarray( [0, 0, 50], dtype = np.uint8 )
	elif np.array_equal( label, slow_left_forward_label ) :
		return np.asarray( [1, 0, 50], dtype = np.uint8 )
	elif np.array_equal( label, slow_right_forward_label ) :
		return np.asarray( [0, 1, 50], dtype = np.uint8 )
	elif np.array_equal( label, fast_forward_label ) :
		return np.asarray( [0, 0, 100], dtype = np.uint8 )
	elif np.array_equal( label, fast_left_forward_label ) :
		return np.asarray( [1, 0, 100], dtype = np.uint8 )
	elif np.array_equal( label, fast_right_forward_label ) :
		return np.asarray( [0, 1, 100], dtype = np.uint8 )
	else :
		return np.asarray( [0, 0, 0], dtype = np.uint8 )
