#!/usr/bin/env python3
# coding=UTF-8

import sys


import os
from enum import Enum, unique
import numpy
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint

from keras import backend

if backend.backend() == 'tensorflow':
	# Set allow_growth on the TF session to be nice to other guys using the cluster I'm training on.
	# The flag causes TF to not fill all the memory available on the GPU, but to allocate it on the fly instead.
	# Also don't forget to check nvidia-smi for free GPUs and restrict the program to those using
	# CUDA_VISIBLE_DEVICES="2" environment variable.
	import tensorflow as tf

	tf_config = tf.ConfigProto(allow_soft_placement=True)
	tf_config.gpu_options.allow_growth = True
	tf_session = tf.Session(config=tf_config)
	backend.set_session(tf_session)


@unique
class Action(Enum):
	TRAIN = 1
	PREDICT = 2

usage_string = "Usage:\n\t%s (--train|--predict) model-file corpus-file\n" % (sys.argv[0])

if len(sys.argv) != 4:
	sys.stderr.write("Error: action, model filename and corpus name required.\n" + usage_string)
	sys.exit(1)

action_name = sys.argv[1]
action = None
if action_name == "--train":
	action = Action.TRAIN
elif action_name == "--predict":
	action = Action.PREDICT
else:
	sys.stderr.write("Error: wrong action specified, please use --train or --predict as the first argument.\n" + usage_string)
	sys.exit(1)

modelname = sys.argv[2]
corpusname = sys.argv[3]

# Load the corpus from the STDIN.
#everything = list(sys.stdin.read())
with open(corpusname, "rb") as f: # Binary mode for cross-compatibility between Python2 and Python3 – in 2, there is no difference and I have to call decode on the returned value. In 3, I can only call decode on bytes, which are returned in binary mode only. Therefore choose binary.
	everything = list(f.read().decode('utf-8'))


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
# How long is a feature-vector?
feature_vec_len = len(fulltext_features[0])
# How wide should each LSTM layer be?
network_width = 256
# Strength of dropout after each LSTM layer.
dropout_strength = 0.05
# The batch size. Since the LSTM is stateful, this is required both for training and for testing.
batch_size = 64
# How many epochs to train for?
nr_epochs = 20
# Whether to train a stateful LSTM or not.
stateful_lstm = False

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
model.add(LSTM(network_width, stateful=stateful_lstm, batch_input_shape=(batch_size, seqlen, 1), return_sequences=True))
model.add(Dropout(dropout_strength))
model.add(LSTM(network_width, stateful=stateful_lstm))
model.add(Dropout(dropout_strength))
model.add(Dense(len(dictchar), activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam")

# Allow checkpointing of an unfinished model after each epoch.
#checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
#callbacks_list = [checkpoint]

if (action == Action.TRAIN):
	model_basename = "%s-%d-dense%d+drop%f+dense%d+drop%f+lstm%d+drop%f+lstm%d+drop%f" % (modelname, seqlen, network_width, dropout_strength, network_width, dropout_strength, network_width, dropout_strength, network_width, dropout_strength)
	
	# Train the model.
	if stateful_lstm:
		for epoch in range(nr_epochs):
			# Other params: callbacks=callbacks_list
			#               initial_epoch=epoch      # Not yet in the version from pip.
			model.fit(dataX, dataY, nb_epoch=1, batch_size=batch_size, shuffle=(not stateful_lstm))
			model.save("%s-%d.h5" % (model_basename, epoch))
			model.reset_states()
	else:
		# Allow checkpointing of an unfinished model after each epoch.
		filepath = model_basename + "-{epoch:02d}-{loss:.4f}.h5"
		checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
		callbacks_list = [checkpoint]
		model.fit(dataX, dataY, nb_epoch=nr_epochs, batch_size=batch_size, shuffle=(not stateful_lstm), callbacks=callbacks_list)
else:
	sys.stderr.write("Unrecognized action error.\n")
	sys.exit(1)
