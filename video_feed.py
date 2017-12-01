import numpy as np
import cv2
import connectivity_manager as cm

from sys import getsizeof

server_address = '192.168.0.102'
port = 5000

capture = cv2.VideoCapture(0)
capture.set(3, 320)
capture.set(4, 240)

connection = cm.set_up_server(server_address, port)

while True :

	if connection.recv(16) > 0 :

		flag, frame = capture.read()
		flat_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()

		data = flat_gray_frame.tostring()
		print 'Sending ' , getsizeof(data) , ' bytes of data ...' 
		connection.sendall(data)

	else :

		print 'Waiting for request ...'


