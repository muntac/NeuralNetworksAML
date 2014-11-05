from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
import numpy as np
import csv
import math
import random

test_inputs = np.load(open('test_inputs.npy','rb'))
train_inputs = np.load(open('train_inputs.npy','rb'))
train_outputs = np.load(open('train_outputs.npy','rb'))

cds = ClassificationDataSet(train_inputs.shape[1], nb_classes=10, class_labels=np.arange(10))
cdsfull = ClassificationDataSet(train_inputs.shape[1], nb_classes=10, class_labels=np.arange(10))
for x in xrange(train_inputs.shape[0]):
	cds.addSample(train_inputs[x], train_outputs[x])
	cdsfull.addSample(train_inputs[x], train_outputs[x])

tstdata, trndata = cds.splitWithProportion(0.25)
trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()
cdsfull._convertToOneOfMany()

fnn = buildNetwork(trndata.indim, 10, trndata.outdim, outclass=SoftmaxLayer)
trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

fnnfull = buildNetwork(cdsfull.indim, 10, cdsfull.outdim, outclass=SoftmaxLayer)
trainerfull = BackpropTrainer(fnnfull, dataset=cdsfull, momentum=0.1, verbose=True, weightdecay=0.01)


for x in xrange(500):
	trainer.trainEpochs(1)
	trnresult = percentError(trainer.testOnClassData(), trndata['class'])
	tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
	with open("PyBrainStatus.txt", "a") as myfile:
	    myfile.write("\n")
	    myfile.write("\nEpoch %4d:"%x)
	    myfile.write("\nTrain Error %f"%trnresult)
	    myfile.write("\nTest Error %f"%tstresult)

	#On test set
	trainerfull.trainEpochs(1)
	strfn = 'PBEpoch%dOutput.csv'%x
	test_output_file = open(strfn, "wb")
	writer = csv.writer(test_output_file, delimiter=',') 
	writer.writerow(['Id', 'Prediction']) # write header
	for y in xrange(test_inputs.shape[0]):
	    res = fnnfull.activate(test_inputs[y])
	    row = [y+1, res.argmax()]
	    writer.writerow(row)
	test_output_file.close()

		



