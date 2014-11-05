import numpy as np
import csv
import random
import math


train_inputs = []
train_outputs = []

print "Loading Training Inputs."
train_inputs = np.load(open("train_inputs.npy", "rb"))
print "Loading Training Outputs."
train_outputs = np.load(open("train_outputs.npy", "rb"))
print "Loading Test Inputs."
test_inputs = np.load(open("test_inputs.npy", "rb"))

trainingsize = train_inputs.shape[0] #50000
#trainingsize = 5000

learningrate = 0.1
epochs = 100
featuredimension = train_inputs.shape[1] #2305 (features + bias)
hiddenunits = 10
outputunits = 10
momentum = 0.1

weights_01 = np.arange(float(featuredimension * hiddenunits)).reshape(featuredimension, hiddenunits) #Weights going from input layer to hidden layer
weights_12 = np.arange(float(hiddenunits * outputunits)).reshape(hiddenunits, outputunits) #Weights going from hidden layer to output layer

out_encoding = np.arange(100).reshape(10,10) #Encoding for how the output should look
out_encoding.fill(0)
#For momentum. For storing the changes in weight from the previous iteration.
prev_deltaw_01 = np.arange(float(featuredimension * hiddenunits)).reshape(featuredimension, hiddenunits) #Previous weights going from input layer to hidden layer
prev_deltaw_12 = np.arange(float(hiddenunits * outputunits)).reshape(hiddenunits, outputunits) #Previous weights going from hidden layer to output layer

prev_deltaw_01.fill(float(0))
prev_deltaw_12.fill(float(0))

#Setting output encoding. Using 1-of-k
for i in xrange(10):
	out_encoding[i,i] = 1

#Initialize weights to random values
for i in xrange(0, hiddenunits):
	for j in xrange(featuredimension):
		weights_01[j,i] = float(random.uniform(-1,1)/1104)
	for j in xrange(outputunits):
		weights_12[i,j] = float(random.uniform(-1,1)/5)

with open("IncrementalTestStatus.txt", "a") as myfile:
    myfile.write("Training...")
    myfile.write("\nLearning rate:%f Momentum:%f Hidden Units:%d Training Size:%d"%(learningrate,momentum,hiddenunits,trainingsize))
for graddesc in xrange(epochs):
	with open("IncrementalTestStatus.txt", "a") as myfile:
	    myfile.write("\n\nEpoch %d" %graddesc)
	error = float(0)
	for example in xrange(trainingsize):
		#1.Compute Output/Forward Pass
		#1a-For Hidden Layer
		out_hiddenunits = train_inputs[example].dot(weights_01)
		out_hiddenunits = 1 / (1 + math.e**(-out_hiddenunits))
		#1b-For Output Layer
		out_outputnodes = out_hiddenunits.dot(weights_12)
		out_outputnodes = 1 / (1 + math.e**(-out_outputnodes))
		#Use SoftMax to transform output
		sumf = np.sum(out_outputnodes)
		out_outputnodes = out_outputnodes / sumf
		#2.Find Error
		#2a-Find error for Output Layer
		realresult = train_outputs[example]
		erroutput = np.arange(float(outputunits))
		for i in xrange(outputunits):
			erroutput[i] = out_outputnodes[i] * (1 - out_outputnodes[i]) * (float(out_encoding[realresult,i]) - out_outputnodes[i])
		
		#2b-Find error for Hidden Layer
		errhiddenunits = np.arange(float(hiddenunits))
		for i in xrange(hiddenunits):
			for j in xrange(outputunits):
				errhiddenunits[i] += out_hiddenunits[i] * (1 - out_hiddenunits[i]) * (weights_12[i,j]) * erroutput[j]

		#3.Update Weights
		#Update weights from hidden to output
		for i in xrange(hiddenunits):
			for j in xrange(outputunits):
				deltawh = (learningrate * erroutput[j] * out_hiddenunits[i]) + (momentum * prev_deltaw_12[i,j])
				prev_deltaw_12[i,j] = deltawh
				weights_12[i,j] = weights_12[i,j] + deltawh


		#Update weights from Hidden Layer to Output Layer
		for i in xrange(0, featuredimension):
			for j in xrange(0, hiddenunits):
				deltaw = (learningrate * errhiddenunits[j] * train_inputs[example,i]) + (momentum * prev_deltaw_01[i,j])
				prev_deltaw_01[i,j] = deltaw
				weights_01[i,j] = weights_01[i,j] + deltaw
		mnerror = float(0)
		for i in xrange(outputunits):
			mnerror += (float(out_encoding[realresult,i]) - out_outputnodes[i])**2
		error += mnerror
		if example % 1000 == 0:
			with open("IncrementalTestStatus.txt", "a") as myfile:
			    myfile.write("\nComplete: %d examples Error: %f"%(example, mnerror))
	strfn = 'Epoch%dweights_01'%graddesc
	np.save(strfn, weights_01)
	strfn = 'Epoch%dweights_12'%graddesc
	np.save(strfn, weights_12)

	with open("IncrementalTestStatus.txt", "a") as myfile:
	    myfile.write("\n...\nError: %f"%error)
	    myfile.write("\nRunning validation")
	test_output = np.arange(test_inputs.shape[0])
	for x in xrange(test_inputs.shape[0]):
		#1.Compute Output/Forward Pass
		#1a-For Hidden Layer
		out_hiddenunits = test_inputs[x].dot(weights_01)
		out_hiddenunits = 1 / (1 + math.e**(-out_hiddenunits))
		#1b-For Output Layer
		out_outputnodes = out_hiddenunits.dot(weights_12)
		out_outputnodes = 1 / (1 + math.e**(-out_outputnodes))
		#Use SoftMax to transform output
		sumf = np.sum(out_outputnodes)
		out_outputnodes = out_outputnodes / sumf
		#print out_outputnodes
		final_output = float(0)
		final_output_idx = 0

		for i in xrange(outputunits):
			if final_output < out_outputnodes[i]:
				final_output = out_outputnodes[i]
				final_output_idx = i
		test_output[x] = final_output_idx
	strfn = 'Epoch%dOutput.csv'%graddesc
	test_output_file = open(strfn, "wb")
	writer = csv.writer(test_output_file, delimiter=',') 
	writer.writerow(['Id', 'Prediction']) # write header
	for x in xrange(test_inputs.shape[0]):
	    row = [x+1, test_output[x]]
	    writer.writerow(row)
	test_output_file.close()

	with open("IncrementalTestStatus.txt", "a") as myfile:
	    myfile.write("\nTesting Completed")
	    
with open("IncrementalTestStatus.txt", "a") as myfile:
	myfile.write("Completed")

del train_inputs
del train_outputs
del test_inputs
