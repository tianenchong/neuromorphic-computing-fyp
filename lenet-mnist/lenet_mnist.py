# USAGE
# python lenet_mnist.py --save-model 1 --weights output/lenet_weights.hdf5
# python lenet_mnist.py --load-model 1 --weights output/lenet_weights.hdf5

# import the necessary packages
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from pyimagesearch.cnn.networks.lenet import LeNet
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.optimizers import SGD, Adadelta
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]

def showImage(testData,prediction = None):
	# extract the image from the testData if using "channels_first"
	# ordering
	if K.image_data_format() == "channels_first":
		image = (testData[0] * 255).astype("uint8")

	# otherwise we are using "channels_last" ordering
	else:
		image = (testData * 255).astype("uint8")

	# merge the channels into one image
	image = cv2.merge([image] * 3)

	# resize the image from a 28 x 28 image to a 96 x 96 image so we
	# can better see it
	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)

	# show the image and prediction
	if prediction is not None:
		cv2.putText(image, str(prediction), (5, 20),
					cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
	cv2.imshow("Digit", image)

def lim(w,mul):
	if w <= 1/(3*mul):
		return 1/(3*mul)
	elif w >= 1/(1*mul):
		return 1/(1*mul)
	else:
		return w

def compdecomp(w,mul):
    compressed = w*(1/(1*mul)-1/(3*mul))+1/(3*mul)
    round = step(compressed, mul)
    decompressed = (round-1/(3*mul))/(1/(1*mul)-1/(3*mul))
    #if w < 0:
    #    print("before")
    #    print(w)
    #    print("after compressed")
    #    print(compressed)
    #    print("after round")
    #    print(round)
    #    print("after decompressed")
    #    print(decompressed)
    return decompressed
	
def comptransfer(w,mul):
    compressed = w*(1/(1*mul)-1/(3*mul))+1/(3*mul)
    round = step(compressed, mul)
    return 1.0/round*1000

def stepMid(w,mul):
    if w <= 1/(3*mul):
        return 1/(3*mul)
    else:
        if w <= 1/(1.5*mul):
            if (w - 1/(3*mul)) < (1/(1.5*mul) - w):
                return 1/(3*mul)
            else:
                return 1/(1.5*mul)
        else:
            if w <= 1/(1*mul):
                if (w - 1/(1.5*mul)) < (1/(1*mul) - w):
                    return 1/(1.5*mul)
                else:
                    return 1/(1*mul)
            else:
                return 1/(1*mul)

def	step(w, mul):
    if w <= 1/(3*mul):
        return 1/(3*mul)
    else:
        if w <= 1/(2.75*mul):
            if (w - 1/(3*mul)) < (1/(2.75*mul) - w):
                return 1/(3*mul)
            else:
                return 1/(2.75*mul)
        else:
            if w <= 1/(2.5*mul):
                if (w - 1/(2.75*mul)) < (1/(2.5*mul) - w):
                    return 1/(2.75*mul)
                else:
                    return 1/(2.5*mul)
            else:
                if w <= 1/(2.25*mul):
                    if (w - 1/(2.5*mul)) < (1/(2.25*mul) - w):
                        return 1/(2.5*mul)
                    else:
                        return 1/(2.25*mul)
                else:
                    if w <= 1/(2*mul):
                        if (w - 1/(2.25*mul)) < (1/(2*mul) - w):
                            return 1/(2.25*mul)
                        else:
                            return 1/(2*mul)
                    else:
                        if w <= 1/(1.75*mul):
                            if (w - 1/(2*mul)) < (1/(1.75*mul) - w):
                                return 1/(2*mul)
                            else:
                                return 1/(1.75*mul)
                        else:
                            if w <= 1/(1.5*mul):
                                if (w - 1/(1.75*mul)) < (1/(1.5*mul) - w):
                                    return 1/(1.75*mul)
                                else:
                                    return 1/(1.5*mul)
                            else:
                                if w <= 1/(1.25*mul):
                                    if (w - 1/(1.5*mul)) < (1/(1.25*mul) - w):
                                        return 1/(1.5*mul)
                                    else:
                                        return 1/(1.25*mul)
                                else:
                                    if w <= 1/(1*mul):
                                        if (w - 1/(1.25*mul)) < (1/(1*mul) - w):
                                            return 1/(1.25*mul)
                                        else:
                                            return 1/(1*mul)
                                    else:
                                        return 1/(1*mul)

def	stepUniform(w, n):
	#range from 0 - n
	return round(w * n)/n
	
	
def modifyWeight(weight,mod,n):
	w = np.array(weight[0])
	woshape = w.shape
	wo = w
	
	tot_num = 1
	for a in w.shape:
		tot_num = tot_num*a
	
	w = np.reshape(w,tot_num)
	
	for i in range(tot_num):
		w[i] = mod(w[i],n)
	weight[0] = np.reshape(w,woshape)
	return weight

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())

# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)
print("[INFO] downloading MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# if we are using "channels first" ordering, then reshape the
# design matrix such that the matrix is:
# num_samples x depth x rows x columns
if K.image_data_format() == "channels_first":
	trainData = trainData.reshape((trainData.shape[0], 1, 28, 28))
	testData = testData.reshape((testData.shape[0], 1, 28, 28))

# otherwise, we are using "channels last" ordering, so the design
# matrix shape should be: num_samples x rows x columns x depth
else:
	trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
	testData = testData.reshape((testData.shape[0], 28, 28, 1))

# scale data to the range of [0, 1]
#trainData = trainData.astype("float32") / (255.0*62.6)
#testData = testData.astype("float32") / (255.0*62.6)
trainData = trainData.astype("float32") / (255.0*255.0)
testData = testData.astype("float32") / (255.0*255.0)
#trainData = trainData.astype("float32") / (255.0*2)
#testData = testData.astype("float32") / (255.0*2)
#trainData = trainData.astype("float32") / 255.0
#testData = testData.astype("float32") / 255.0

# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

# initialize the optimizer and model
weightsPath=args["weights"] if args["load_model"] > 0 else None
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(numChannels=1, imgRows=28, imgCols=28,
	numClasses=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
	


# only train and evaluate the model if we *are not* loading a
# pre-existing model
if args["load_model"] < 0:
	print("[INFO] training...")
	model.fit(trainData, trainLabels, batch_size=128, epochs=100,
		verbose=1)

        
#showImage(testData[9990],7)
#cv2.waitKey(0)
#showImage(testData[9999],6)
#cv2.waitKey(0)

# show the accuracy on the testing set
if args["load_model"] < 0:
	print("[INFO] evaluating...")
	(loss, accuracy) = model.evaluate(testData, testLabels,
		batch_size=128, verbose=1)
	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
elif weightsPath is not None:
	model.load_weights(weightsPath)
	print(model.summary())
	print("[INFO] evaluating...")
	accs = []

	r = 100
	layerweights = {}
	max = 0
	maxlocs = []
	wantedlayers = [3]
	allrange = 1
	transfer = 1
	originalw = 1
	justone = 1
	outputlayer = 0
	#func = comptransfer
	#func = compdecomp
	#func = step
	#func = stepUniform
	func = lim
    
	layernum = len(model.layers)
	for l in range(layernum):
		w = model.layers[l].get_weights()
		if len(w) > 0:
			layerweights[l] = w
			#print("original layer "+str(l))
			#print(w)
	if transfer == 1:
		f= open("weight_original","w+")
		for l in layerweights:
				layermweight = np.copy(layerweights[l])
				if originalw != 1:
					modifyWeight(layermweight,func,1)
				model.layers[l].set_weights(layermweight)
				w = model.layers[l].get_weights()
				if len(w) > 0:
					if originalw == 1:
						print("layer "+str(l))
					else:
						print("modified layer "+str(l))
					print(w)
					if l in wantedlayers:
						w_transposed = (np.asarray(w)).transpose()
						for a in w_transposed:
							for b in a:
								f.write(to_str(b))
								f.write('\n')
		f.close()
	else:
		if allrange == 1:
			for i in range(r):
				for l in layerweights:
					layermweight = np.copy(layerweights[l])
					modifyWeight(layermweight,func,i+1)
					model.layers[l].set_weights(layermweight)
				
				(loss, acc) = model.evaluate(testData, testLabels,
					batch_size=128, verbose=1)
				print("[{:3d}] accuracy: {:.2f}%".format(i+1,acc * 100))
				if max <= acc * 100:
					if max < acc * 100:
						max = acc * 100
						maxlocs.clear()
						maxlocs.append(i+1)
					else:
						maxlocs.append(i+1)

				accs.append(acc*100)
			print('Max value: {:.2f}% at location(s): [{:}]'.format(max,','.join(map(str, maxlocs))))
			print('value at location 8: {:.2f}%'.format(accs[7]))
			fig = plt.figure(1)
			plot = fig.add_subplot(111)
			plot.tick_params(axis='both', labelsize=16)
			plt.plot(range(1,r+1),accs)
			plt.ylim(0, 100)
			plt.ylabel('Accuracy %', fontsize=12)
			if func == stepUniform:
				plt.xlabel('Steps', fontsize=12)
			else:
				plt.xlabel('Scaled R (Ohm)', fontsize=12)
			plt.show()
		else:
			for l in layerweights:
				layermweight = np.copy(layerweights[l])
				modifyWeight(layermweight,func,1)
				model.layers[l].set_weights(layermweight)
				w = model.layers[l].get_weights()
				#if len(w) > 0:
					#print("modified layer "+str(l))
				w = np.array(w)
				print("layer "+str(l))
				print(w.shape)
			
			print("Output:")
			
			inp = model.input
				
			outputs = [layer.output for layer in model.layers]

			fun = K.function(inp,outputs)# evaluation function
			f= open("intermediate","w+")
			testnum = len(testLabels)
			testnum = (1 if justone == 1 else testnum)
			for i in range(testnum):
				layer_outputs = fun(testData[np.newaxis, i])
				j = 0
				
				for layer_output in layer_outputs:
					out = np.array(layer_output)
					print("layer "+str(j))
					print(out.shape)
					#if j == outputlayer:
					#	if justone ==0:
					#		f.write("{:d} ".format(int(np.argmax(testLabels[i]))))
					#		for a in layer_output:
					#			for b in a:
					#				f.write(to_str(b))
					#				f.write(' ')
					#			if i != testnum-1:
					#				f.write('\n')
					#	else:
					#		for a in layer_output:
					#			for b in a:
					#				f.write(to_str(b))
					#				f.write('\n')
					j = j+1
			f.close()
			
			#for layer_output in layer_outputs:
			#	if i == 1:
			#		f= open("intermediate","w+")
			#		for a in layer_output:
			#			for b in a:
			#				f.write(to_str(b))
			#				f.write(' ')
			#			f.write('\n')
			#		f.close()
			#	print(layer_output)#printing the outputs of layers
			#	i = i+1
			#print(layer_outputs[-1].argmax(axis=1))
	
#weightsPath=args["weights"] if args["load_model"] > 0 else None
#if weightsPath is not None:
	#model.load_weights(weightsPath)
# check to see if the model should be saved to file
if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"], overwrite=True)

# randomly select a few testing digits
#if args["load_model"] < 0:
#	for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
		# classify the digit
#		probs = model.predict(testData[np.newaxis, i])
#		prediction = probs.argmax(axis=1)

#		showImage(testData[i],prediction[0])
#		print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
#		np.argmax(testLabels[i])))
#		cv2.waitKey(0)