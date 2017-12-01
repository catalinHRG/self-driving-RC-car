import serial as sr
import numpy as np
from keyboard import is_pressed

def get_steering_vector(forward_key, left_key, right_key) :

	return np.asarray( [ is_pressed(forward_key), is_pressed(left_key), is_pressed(right_key) ] ).astype(np.uint8)


usb_device = '/dev/ttyUSB0'
baud_rate = 115200

arduino_link = sr.Serial(usb_device, baud_rate)

while True :

	steering = get_steering_vector('up', 'left', 'right')

	string = ''

	for i in range ( len(steering) ) : 

		string += str(steering[i])

	print 'Sending ', string
	arduino_link.write( string.encode('ascii') )