# self-driving-RC-car

# -------------------------------------------

# RPI3 mounted on the RC car alongside an USB camera and a power bank. The added weight will come in handy since it will limit the speed of the RC car to a more managable amount given the current overall delay on the communication chain (network + computation latency)

# The raspberry pi will do the capture and some pre processing before sending via TCP/IP socket the bottom half of the flattened frame matrix to a laptop.
# The bytes are collected, decoded, and fed into the TensorFlow model where all the computation is done on the GPU.
# The resulting steering vector is sent back to the rpi 3 via socket so that it can be used to control the motors using the GPIO pins. The speed at which the car will go forward is controlled simply by changing the freq / duty cycle of a pwm signal.

# -------------------------------------------

# When it comes to collecting the data set needed to train the model, the RC car is being droven using the left, right arrow keys 
# on the keyboard while on the laptop frames are being streamed over provinding a first person view. 
# The sample / label pairs are being stored into an hdf5 file in real time.

# --------------------------------------------

# Multilayer Perceptron with a unidimensional vector comprised of 38400 of pixel values as input, 
# one hidden layer consisting of 150 units, 7 units in the output layer which will add up to create a one-hot encoded vector, leading
# to 7 possible classes : stopped, slow forward, slow forward steering left, slow forward steering right and the forward, left, right equivalent for high speed.




