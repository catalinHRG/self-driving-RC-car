'''

This module will collect the sample/label pairs and store them on disk into an hdf5 file while the RC car is being 
diven down the track. The driver has a first person view from the camera mounted on the RC car and all he has to do
is to use the arrow keys on the keyboard to control the car.

'''

import numpy as np
import cv2
import socket as sk
import h5py as h5
import sys

from utils import data_set_manager as dsm
from utils import connectivity_manager as cm
from time import sleep, time
from keyboard import is_pressed

def get_steering_vector(left_key, right_key, increase_key, decrease_key, delta, duty_cycle) :

	if is_pressed( increase_key ) :
		if duty_cycle < 100 :
			duty_cycle = duty_cycle + delta
	elif is_pressed( decrease_key ) :
		if duty_cycle > 0 :
			duty_cycle = duty_cycle - delta

	temp = np.asarray( [ is_pressed(left_key), is_pressed(right_key) ] )
	temp = np.append( temp, duty_cycle )

	return temp.astype(np.uint8), duty_cycle

# global variables
dsm_writer = dsm.HDF5_Writer('utils/DataSet.h5')

client_ip_address = sys.argv[1]
port = int( sys.argv[2] )

tr_set_size = dsm_writer.get_tr_set_size()
te_set_size = dsm_writer.get_te_set_size()
sample_size = dsm_writer.get_sample_size()
label_size = dsm_writer.get_label_size()

index_pointer = counter = 0

total_bytes_per_frame = 76821 # 320 X 240 uin8 pixels values
bytes_per_chunk = 4096

client_sk = cm.connect_to_server(client_ip_address, port)

steering_vector = np.asarray( [0, 0, 0] ).astype(np.uint8)
speed = 0

data_set = 'train'
flag = True

while counter < total :

	print 'Sending steering vector : ', steering_vector
	client_sk.sendall( steering_vector.tostring() )

	#start = time()
	raw_data = cm.collect_bytes(client_sk, total_bytes_per_frame, bytes_per_chunk)
	#print 'Collected bytes in ... ', time() - start

	frame = np.fromstring( raw_data, dtype = np.uint8 )
	frame = np.reshape( frame, [240, 320] )

	cv2.waitKey( 1 )
	cv2.imshow('Video', frame)

	steering_vector, speed = get_steering_vector('left', 'right', 'q', 'w', 50, speed)

	label = dsm.convert_to_label( steering_vector )

	bottom_half = ( np.split( frame.flatten(), 2 ) ) [1]

	if flag :
		if counter > tr_set_size :
			data_set = 'test'
			flag = False

	dsm_writer.insert_entry(bottom_half, label, index_pointer, data_set)

	index_pointer += 1
	if index_pointer == tr_set_size : index_pointer = 0

	counter += 1

dsm_writer.close_file()
cv2.destroyAllWindows()

