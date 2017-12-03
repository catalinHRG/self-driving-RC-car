import socket as sk

from time import sleep
from sys import getsizeof

def connect_to_server(server_address, port) :

	client = sk.socket(sk.AF_INET, sk.SOCK_STREAM)

	print 'Establishing TCP/IP communication link with the server ...'
	client.connect( (server_address, port) )

	return client

def set_up_server(server_address, port) :

	server = sk.socket(sk.AF_INET, sk.SOCK_STREAM)

	print 'Binding address / port to socket ...'
	server.bind((address, port))
	server.listen(0)

	print 'Waiting for connection ...'
	connection, client_address = server.accept()
	print 'Connection established on address ' , client_address[0] , ' port ' , client_address[1]

	sleep(2)

	return connection

def collect_bytes(client, total, chunk_size) :

	data = bytes()

	while getsizeof(data) <= total :

		bucket = client.recv(chunk_size)
		#print 'so far we got ', getsizeof(data)
		data += bucket

	return data
