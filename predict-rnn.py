#!/usr/bin/env python3

import sys

if len(sys.argv) != 2:
	sys.stderr.write("Error: filename for model loading required.\nUsage:\n\t%s model-file\n" % (sys.argv[0]))
	sys.exit(1)


import numpy
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation


# Load the corpus from the STDIN.
everything = sys.stdin.read()


# Find all the character types occuring in the corpus.
allchars = sorted(list(set(everything)))

# A dictionary which contains a mapping from chars to numbers. Each char gets its own unique number.
chardict = dict(zip(allchars, range(0, len(allchars))))
# Reverse mapping from numbers to chars.
dictchar = dict(zip(chardict.values(), chardict.keys()))

# Convert the corpus from chars to character numbers.
fulltext = [chardict[ch] for ch in list(everything)]



# How many previous chars to supply as features for training?
seqlen = 32

# Convert the corpus to features.
dataX = []
for i in range(0, len(fulltext) - seqlen, 1):
	dataX.append(fulltext[i:i+seqlen])


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

model.load_weights(sys.argv[1])



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
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print("\nDone.")
