# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.constraints import NonNegStep, NonNeg, MinMaxNorm
from keras import backend as K
import numpy as np

class LeNet:
	@staticmethod
	def build(numChannels, imgRows, imgCols, numClasses,
		activation="relu"):
		# initialize the model
		model = Sequential()
		inputShape = (imgRows, imgCols, numChannels)
		constraint = MinMaxNorm(min_value=0.0, max_value=1.0,rate=1.0,axis=0)
		#constraint = NonNeg()
		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (numChannels, imgRows, imgCols)

		# define the first set of CONV => ACTIVATION => POOL layers
		#model.add(Conv2D(20, 5, padding="same",use_bias=False,kernel_constraint=constraint, input_shape=inputShape))
		#model.add(Activation(activation))
		#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# define the second set of CONV => ACTIVATION => POOL layers
		#model.add(Conv2D(20, 5, padding="same",use_bias=False,kernel_constraint=constraint))
		#model.add(Activation(activation))
		#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# define the first FC => ACTIVATION layers
		model.add(Flatten(input_shape=inputShape))
		model.add(Dense(20,use_bias=False,kernel_constraint=constraint))
		model.add(Activation(activation))
		#model.add(Activation("softmax"))
		# define the second FC layer
		model.add(Dense(numClasses,use_bias=False,kernel_constraint=constraint))

		# lastly, define the soft-max classifier
		#model.add(Activation(activation))
		model.add(Activation("softmax"))

		# if a weights path is supplied (inicating that the model was
		# pre-trained), then load the weights
			
		# return the constructed network architecture
		return model