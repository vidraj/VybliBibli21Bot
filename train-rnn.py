#!/usr/bin/env python3

import sys

if len(sys.argv) != 2:
	sys.stderr.write("Error: filename for model saving required.\nUsage:\n\t%s model-file\n" % (sys.argv[0]))
	sys.exit(1)


import os
import numpy
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint


# Load the corpus from the STDIN.
everything = sys.stdin.read()


# Find all the character types occuring in the corpus. Also serves as mapping from char-IDs to chars
dictchar = sorted(list(set(everything)))

# A dictionary which contains a mapping from chars to numbers. Each char gets its own unique number.
chardict = dict(zip(dictchar, range(0, len(dictchar))))

# Convert the corpus from chars to character numbers.
fulltext = [chardict[ch] for ch in list(everything)]

# The cleartext is no longer needed.
del(everything)


# How many previous chars to supply as features for training?
seqlen = 32

# Convert the corpus to features.
dataX = []
dataY = []
for i in range(0, len(fulltext) - seqlen, 1):
	dataX.append(fulltext[i:i+seqlen])
	dataY.append(fulltext[i+seqlen])

# The output value should be one-hot encoded.
y = to_categorical(dataY)

# The input values have to be reoriented; they are expressed as floats in [0, 1].
x = numpy.reshape(dataX, (len(dataX), seqlen, 1))
x = x/len(chardict)


# Create the Keras model.
model = Sequential()
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam")

# Allow checkpointing of an unfinished model after each epoch.
filepath="vybli-checkpoint-32-lstm256+drop0.2+lstm256+drop0.2-{epoch:02d}-{loss:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Train the model.
model.fit(x, y, nb_epoch=50, batch_size=128, callbacks=callbacks_list)

# Save the result.
model.save(sys.argv[1])














# Create a single test data point and convert it into numbers in the correct format.
candidate = numpy.reshape([chardict['b'], chardict['y'], chardict['l']], (1,3,1))
# Convert into floats.
candidate = candidate/len(chardict)
# Predict the next char probability distribution.
prediction = model.predict(candidate)

#prediction
#sum(sum(prediction))
# -> ~1

# Retrieve the most likely next char.
answer = numpy.argmax(prediction)
print("‘byl’ predicts ‘%s’" % dictchar[answer])


# Do the same for another datapoint.
candidate2 = numpy.reshape([chardict['H'], chardict['o'], chardict['s']], (1,3,1))
#candidate2
candidate2 = candidate2/len(chardict)
prediction2 = model.predict(candidate2)
#prediction2[0][chardict['-']]
#prediction2[0][chardict['a']]
print("‘Hos’ predicts ‘%s’" % dictchar[numpy.argmax(prediction2)])

#candidate3 = numpy.reshape([chardict['a'], chardict['?'], chardict['-']], (1,3,1))
#candidate3 = candidate3/len(chardict)
#candidates = numpy.array([candidate2,candidate3])
##candidates.shape
#candidates2 = candidates.reshape(2,3,1)
##candidates2
#predictions = model.predict(candidates2)
#numpy.argmax(predictions[0])
#numpy.argmax(predictions[1])


## Generate new text.

# Pick a random seed to kick the process off.
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed: “%s”" % (''.join([dictchar[value] for value in pattern])))
# Generate characters starting with that seed.
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(len(chardict))
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = dictchar[index]
	#seq_in = [dictchar[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print("\nDone.")










