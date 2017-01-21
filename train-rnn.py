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
seqlen = 6
# How wide should each LSTM layer be?
network_width = 256
# Strength of dropout after each LSTM layer.
dropout_strength = 0.05
# The batch size. Since the LSTM is stateful, this is required both for training and for testing.
batch_size = 64
# How many epochs to train for?
nr_epochs = 20

# The corpus size must be a multiple of batchsize for stateful training.
#rounded_fulltext_size = (len(fulltext) // batch_size) * batch_size
#fulltext = fulltext[:rounded_fulltext_size + seqlen - 1]

# Convert the corpus to features.
dataX = []
dataY = []
for i in range(0, ((len(fulltext) - seqlen) // batch_size) * batch_size, 1):
	dataX.append(fulltext[i:i+seqlen])
	dataY.append(fulltext[i+seqlen])

# The output value should be one-hot encoded.
y = to_categorical(dataY)

# The input values have to be reoriented; they are expressed as floats in [0, 1].
x = numpy.reshape(dataX, (len(dataX), seqlen, 1))
x = x/len(dictchar)


# Create the Keras model.
model = Sequential()
model.add(LSTM(network_width, stateful=True, batch_input_shape=(batch_size, seqlen, 1), return_sequences=True))
model.add(Dropout(dropout_strength))
model.add(LSTM(network_width, stateful=True))
model.add(Dropout(dropout_strength))
model.add(Dense(len(dictchar), activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam")

# Allow checkpointing of an unfinished model after each epoch.
#checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
#callbacks_list = [checkpoint]

# Train the model.
for epoch in range(nr_epochs):
	# Other params: callbacks=callbacks_list
	#               initial_epoch=epoch      # Not yet in the version from pip.
	model.fit(x, y, nb_epoch=1, batch_size=batch_size, shuffle=False)
	model.save("%s-%d-lstm%d+drop%f+lstm%d+drop%f-%d.h5" % (sys.argv[1], seqlen, network_width, dropout_strength, network_width, dropout_strength, epoch))
	model.reset_states()
