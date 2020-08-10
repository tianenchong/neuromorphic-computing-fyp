import numpy as np
import copy 
from math import exp
from matplotlib import pyplot as plt
from PIL import Image
from random import choice, uniform
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.optimizers import SGD, Adadelta
from keras.utils import np_utils
from keras import backend as K
import argparse
import cv2
import matplotlib.pyplot as plt

matrix = []
for i in open('resreport.txt'):
    matrix.append( list(map(int, i.split())) )


((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

testData = testData.astype("float32") / 255.0

# set the layout for illustration of the system at all 40 T values to 8 by 5
rowsize = 8
colsize = 5
fig,ax=plt.subplots(nrows=rowsize,ncols=colsize,figsize=(20,20)) # figures formatting and cosmetics
fig.tight_layout(pad=3.0) # figures formatting and cosmetics

r = (255, 50, 50)
g = (0, 255, 0)
start = 10000-rowsize*colsize
# show illustration of the system at all 40 T values
for i in range(start,10000):
	if K.image_data_format() == "channels_first":
		image = (testData[i][0] * 255).astype("uint8")

	# otherwise we are using "channels_last" ordering
	else:
		image = (testData[i] * 255).astype("uint8")
	
	# merge the channels into one image
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	prediction = matrix[i][1]
	if prediction is not None:
		cv2.putText(image, str(prediction), (5, 20),
					cv2.FONT_HERSHEY_SIMPLEX, 0.75, (g if matrix[i][0] == matrix[i][1] else r), 2)
	i = i - start
	ax[int(i/colsize), i%colsize].get_yaxis().set_visible(False)
	ax[int(i/colsize), i%colsize].get_xaxis().set_visible(False)
	ax[int(i/colsize)][i%colsize].imshow(image,cmap='gray',vmin=0,vmax=255,aspect='equal',interpolation='nearest')
plt.show()