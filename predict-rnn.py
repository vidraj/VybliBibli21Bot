#!/usr/bin/env python3

import sys

if len(sys.argv) != 2:
	sys.stderr.write("Error: filename for model loading required.\nUsage:\n\t%s model-file\n" % (sys.argv[0]))
	sys.exit(1)


import numpy
from keras.models import load_model


# Load the corpus from the STDIN.
everything = sys.stdin.read()


# Find all the character types occuring in the corpus. Also serves as mapping from char-IDs to chars
dictchar = sorted(list(set(everything)))

# A dictionary which contains a mapping from chars to numbers. Each char gets its own unique number.
chardict = dict(zip(dictchar, range(0, len(dictchar))))

# Convert the corpus from chars to character numbers.
fulltext = [chardict[ch] for ch in list(everything)]



# How many previous chars to supply as features for training?
seqlen = 6
# How many feature vectors to start with?
batch_size = 64

# Convert the corpus to features.
dataX = []
for i in range(0, len(fulltext) - seqlen, 1):
	dataX.append(fulltext[i:i+seqlen])


# The input values have to be reoriented; they are expressed as floats in [0, 1].
x = numpy.reshape(dataX, (len(dataX), seqlen, 1))
x = x/len(chardict)


# Load the Keras model.
model = load_model(sys.argv[1])
#model = Sequential()
#model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(256))
#model.add(Dropout(0.2))
#model.add(Dense(len(chardict), activation="softmax"))
#model.load_weights(sys.argv[1])
#model.compile(loss="categorical_crossentropy", optimizer="adam")




## Generate new text.

for tweet in range(20):
	# Pick a random seed to kick the process off.
	start = numpy.random.randint(0, len(dataX) - batch_size - 1)
	x = dataX[start:start + batch_size]
	last_window = x[-1]
	print("\n\nSeed ends with: “%s”" % (''.join([dictchar[value] for value in last_window])))
	x = numpy.reshape(x, (batch_size, seqlen, 1))
	x = x / float(len(chardict))

	# Generate characters starting with that seed.
	generated_text = ""

	for i in range(256):
		#sys.stdout.write('.')
		#sys.stdout.flush()
		predictions = model.predict_on_batch(x) # , verbose=0
		index = numpy.random.choice(len(chardict), p=predictions[-1])
		generated_text += dictchar[index]

		last_window.append(index)
		last_window = last_window[1:len(last_window)]
		nx = numpy.reshape(last_window, (1, len(last_window), 1))
		nx = nx / float(len(chardict))
		x = numpy.concatenate((x, nx))[1:]
		#print("Our corpus is: ", x)

	print("Generated text: “%s”" % generated_text)
	model.reset_states()
