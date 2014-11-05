import numpy as np
import csv


# Load all training inputs to a python list
train_inputs = []
with open('train_inputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for train_input in reader: 
        train_input_no_id = []
	train_input_no_id.append(float(1))
        for dimension in train_input[1:]:
            train_input_no_id.append(float(dimension))
        train_inputs.append(np.asarray(train_input_no_id)) # Load each sample as a numpy array, which is appened to the python list

train_inputs_np = np.asarray(train_inputs)
np.save('train_inputs', train_inputs_np)
del train_inputs_np

# Load all training ouputs to a python list
train_outputs = []
with open('train_outputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for train_output in reader:  
        train_output_no_id = int(train_output[1])
        train_outputs.append(train_output_no_id)

train_outputs_np = np.asarray(train_outputs)
np.save('train_outputs', train_outputs_np)
del train_outputs_np

# Load all test inputs to a python list 
test_inputs = []
with open('test_inputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for test_input in reader: 
        test_input_no_id = []
	test_input_no_id.append(float(1))
        for dimension in test_input[1:]:
            test_input_no_id.append(float(dimension))
        test_inputs.append(np.asarray(test_input_no_id)) # Load each sample as a numpy array, which is appened to the python list

test_inputs_np=np.asarray(test_inputs)
np.save('test_inputs', test_inputs_np)
del test_inputs_np
# Convert python lists to numpy arrays

# Save as numpy array files

