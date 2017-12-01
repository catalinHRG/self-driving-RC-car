import numpy as np
import cv2
import connectivity_manager as cm
import RPi.GPIO as gpio

from sys import getsizeof

m1_pin1 = 21
m1_pin2 = 22
m2_pin1 = 23
m2_pin2 = 24
pwm_pin1 = 12
# in case we would like to command the steering motor to control the steering angle pwm_pin2 = 0 

initial_frequency = 100
initial_duty_cycle = 0

gpio.setmode(gpio.BOARD)
gpio.setup(m1_pin1, gpio.OUT)
gpio.setup(m1_pin2, gpio.OUT)
gpio.setup(m2_pin1, gpio.OUT)
gpio.setup(m2_pin1, gpio.OUT)
gpio.setup(pwm_pin1, gpio.OUT)
# gpio.setup(pwm_pin2, gpio.OUT)

steering_vector_size = 40 # bytes

server_address = '192.168.0.102'
port = 5000

capture = cv2.VideoCapture(0)
capture.set(3, 320)
capture.set(4, 240)

connection = cm.set_up_server( server_address, port )

steering_vector = np.asarray( [0, 0, 0], dtype = np.uint8 )

# pwm_handler = gpio.PWM( pwm_pie2, initial_frequency )
# pwm_handler.start( initial_duty_cycle )

pwm_handler = gpio.PWM( pwm_pin1, initial_frequency )
pwm_handler.start(initial_duty_cycle)

def apply_sv( steering_vector ) :

	# steering vector -- > [ left -> 1/0, right -> 1/0, speed -> 0/50/100 ]

	gpio.output( m1_pin1, steering_vector[0] )
	gpio.output( m1_pin2, steering_vector[1] )
	pwm_handler.ChangeDutyCycle( steering_vector[2] ) # controlling the actual speed of the car, 0 means stopped

while True :

	raw_data = connection.recv( steering_vector_size )
	steering_vector = np.fromstring( raw_data, dtype = np.uint8 )

	apply_sv( steering_vector )

	flag, frame = capture.read()
	flat_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()

	raw_data = flat_gray_frame.tostring()
	print 'Sending ' , getsizeof(data) , ' bytes of data ...' 
	connection.sendall( raw_data )
