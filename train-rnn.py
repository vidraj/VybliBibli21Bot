#!/usr/bin/env python3
# coding=UTF-8

import sys


import os
from enum import Enum, unique
import numpy
import re
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed
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



class fullprint:
	'context manager for printing full numpy arrays'

	def __init__(self, **kwargs):
		if 'threshold' not in kwargs:
			kwargs['threshold'] = numpy.nan
		self.opt = kwargs

	def __enter__(self):
		self._opt = numpy.get_printoptions()
		numpy.set_printoptions(**self.opt)

	def __exit__(self, type, value, traceback):
		numpy.set_printoptions(**self._opt)




# Load the corpus from the STDIN.
#fulltext = list(sys.stdin.read())
with open(corpusname, "rb") as f: # Binary mode for cross-compatibility between Python2 and Python3 – in 2, there is no difference and I have to call decode on the returned value. In 3, I can only call decode on bytes, which are returned in binary mode only. Therefore choose binary.
	fulltext = list(f.read().decode('utf-8'))


# Find all the character types occuring in the corpus. Also serves as mapping from char-IDs to chars
dictchar = sorted(list(set(fulltext)))

# A dictionary which contains a mapping from chars to numbers. Each char gets its own unique number.
chardict = dict(zip(dictchar, range(len(dictchar))))

vowels = set()
consonants = set()
numerals = set()
punctuations = set()

for char in dictchar:
	if char.isupper():
		# Don't process uppercase. This is potentially bad, because down- and upcasing
		#  are not necessarily reversible. It is in Czech and English, though, which
		#  is what I care about.
		char = char.lower()

	if re.match(u"^[aáeéěiíyýoóuúů]$", char):
		# A vowel.
		vowels.add(char)
	elif char.isalpha():
		# An alphanumeric char that is not a vowel.
		consonants.add(char)
	elif char.isdigit(): # .isdecimal() doesn't exist in Python2
		# A number.
		numerals.add(char)
	else:
		# Anything else gets classified as punctuation, including spaces, newlines and whatnot.
		punctuations.add(char)

#print("Vowels: ", vowels, "\nConsonants: ", consonants, "\nNumerals: ", numerals, "\nOther stuff: ", punctuations)

def make_dict(charlist):
	return dict(zip(sorted(list(charlist)), map(lambda x: float(x+1)/len(charlist), range(len(charlist)))))

vowel_dict = make_dict(vowels)
consonant_dict = make_dict(consonants)
numeral_dict = make_dict(numerals)
punctuation_dict = make_dict(punctuations)


## TODO test of predictions
#vowel_dict = {u'a': 0.07142857142857142, u'\xe1': 0.5, u'e': 0.14285714285714285, u'i': 0.21428571428571427, u'\xed': 0.6428571428571429, u'o': 0.2857142857142857, u'\xe9': 0.5714285714285714, u'\xf3': 0.7142857142857143, u'u': 0.35714285714285715, u'\u016f': 1.0, u'y': 0.42857142857142855, u'\u011b': 0.9285714285714286, u'\xfa': 0.7857142857142857, u'\xfd': 0.8571428571428571}
#consonant_dict = {u'\u010d': 0.7692307692307693, u'\u010f': 0.8076923076923077, u'\u0148': 0.8461538461538461, u'\u0159': 0.8846153846153846, u'\u0161': 0.9230769230769231, u'c': 0.07692307692307693, u'b': 0.038461538461538464, u'\u0165': 0.9615384615384616, u'd': 0.11538461538461539, u'g': 0.19230769230769232, u'f': 0.15384615384615385, u'h': 0.23076923076923078, u'k': 0.3076923076923077, u'j': 0.2692307692307692, u'm': 0.38461538461538464, u'l': 0.34615384615384615, u'n': 0.4230769230769231, u'q': 0.5, u'p': 0.46153846153846156, u's': 0.5769230769230769, u'r': 0.5384615384615384, u't': 0.6153846153846154, u'v': 0.6538461538461539, u'x': 0.6923076923076923, u'z': 0.7307692307692307, u'\u017e': 1.0}
#numeral_dict = {u'1': 0.2, u'0': 0.1, u'3': 0.4, u'2': 0.3, u'5': 0.6, u'4': 0.5, u'7': 0.8, u'6': 0.7, u'9': 1.0, u'8': 0.9}
#punctuation_dict = {u'!': 0.1875, u' ': 0.125, u'"': 0.25, u"'": 0.3125, u')': 0.4375, u'(': 0.375, u'\n': 0.0625, u'-': 0.5625, u',': 0.5, u'.': 0.625, u'\u201a': 1.0, u';': 0.75, u':': 0.6875, u']': 0.9375, u'[': 0.875, u'?': 0.8125}
##dictchar = [u'\n', u' ', u'!', u"'", u'(', u')', u',', u'-', u'.', u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u':', u';', u'?', u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'I', u'J', u'K', u'L', u'M', u'N', u'O', u'P', u'Q', u'R', u'S', u'T', u'U', u'V', u'W', u'Y', u'Z', u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm', u'n', u'o', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y', u'z']
#dictchar = [u'\n', u' ', u'!', u'"', u"'", u'(', u')', u',', u'-', u'.', u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u':', u';', u'?', u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'I', u'J', u'K', u'L', u'M', u'N', u'O', u'P', u'Q', u'R', u'S', u'T', u'U', u'V', u'X', u'Y', u'Z', u'[', u']', u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm', u'n', u'o', u'p', u'r', u's', u't', u'u', u'v', u'x', u'y', u'z', u'\xc1', u'\xc9', u'\xcd', u'\xd3', u'\xda', u'\xdd', u'\xe1', u'\xe9', u'\xed', u'\xf3', u'\xfa', u'\xfd', u'\u010c', u'\u010d', u'\u010e', u'\u010f', u'\u011a', u'\u011b', u'\u0148', u'\u0158', u'\u0159', u'\u0160', u'\u0161', u'\u0165', u'\u016e', u'\u016f', u'\u017d', u'\u017e', u'\u201a']
#chardict = dict(zip(dictchar, range(len(dictchar))))

#print("Vowel dict: ", sorted(vowel_dict, key=vowel_dict.get), "\nConsonant dict: ", sorted(consonant_dict, key=consonant_dict.get), "\nNumeral dict: ", sorted(numeral_dict, key=numeral_dict.get), "\nOther stuff dict: ", sorted(punctuation_dict, key=punctuation_dict.get), "\nDictchar: ", dictchar)

def char_to_feature(char):
	#return [chardict[char]/len(chardict)]
	#return to_categorical([chardict[char]], nb_classes=len(dictchar))[0]
	
	#assert(len(char) == 1)
	lc_form = char.lower()

	return [float(char.isupper()), vowel_dict.get(lc_form, 0.0), consonant_dict.get(lc_form, 0.0), numeral_dict.get(char, 0.0), punctuation_dict.get(char, 0.0)]

def char_to_output(char):
	return to_categorical([chardict[char]], nb_classes=len(dictchar))[0]

def char_vec_to_feature(v):
	return list(map(char_to_feature, v))

def char_vec_to_output(v):
	return to_categorical([chardict[char] for char in v], nb_classes=len(dictchar))

def sample(a, temperature=1.0):
	# helper function to sample an index from a probability array
	a = numpy.log(a) / temperature
	a = numpy.exp(a) / numpy.sum(numpy.exp(a))
	return numpy.random.choice(len(a), p=a)

def output_to_char(o):
	index = sample(o, 0.7) #numpy.random.choice(len(chardict), p=o)
	return dictchar[index]

def output_vec_to_char(v):
	return list(map(output_to_char, v))


# Convert the corpus from chars to character features.
fulltext_features = char_vec_to_feature(fulltext)


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


# Convert the corpus to features.
dataX = []
dataY = []
# The corpus size must be a multiple of batchsize for stateful training.
for i in range(0, ((len(fulltext) - seqlen) // batch_size) * batch_size, 1):
	dataX.append(fulltext_features[i:i+seqlen])
	dataY.append(char_to_output(fulltext[i+seqlen]))

# The input values have to be reshaped into a 3D tensor.
#dataX = numpy.reshape(dataX, (len(dataX), seqlen, feature_vec_len))
dataX = numpy.array(dataX)
dataY = numpy.array(dataY)
#print("Shape of dataX: ", dataX.shape)

#print(dataX)
#print(dataY)

#sys.exit(1)

# Create the Keras model.
model = Sequential()
model.add(LSTM(network_width, stateful=stateful_lstm, batch_input_shape=(batch_size, seqlen, feature_vec_len), return_sequences=True))
model.add(Dropout(dropout_strength))
model.add(LSTM(network_width, stateful=stateful_lstm))
model.add(Dropout(dropout_strength))
model.add(Dense(len(dictchar), activation="softmax"))


# Allow checkpointing of an unfinished model after each epoch.
#checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
#callbacks_list = [checkpoint]

if (action == Action.TRAIN):
	model.compile(loss="categorical_crossentropy", optimizer="adam")
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
		
elif (action == Action.PREDICT):
	# Beware that model files are not portable across platforms, while weights are.
	#  But loading just the weights requires us to manually reconstruct the model layout,
	#  while loading the model does not.
	#  Choose whatever you want here.
	#model.load_weights(modelname)
	#model.compile(loss="categorical_crossentropy", optimizer="adam")
	model = load_model(modelname)
	
	init_len = max(128, batch_size) # This length at the start of out buffer will be used as a seed to initialize the LSTM memory.
	predict_len = max(1024, batch_size) # 140 is the target tweet size, the rest is there to create a larger possibility of finding a meaningful tweet.
	total_len = init_len + predict_len # TODO make sure this length is divisible by batch_size
	
	# Tweet every 10 minutes for 5 hours → 31 tweets.
	for tweet in range(31):
		# Create the data array. First part will be prefilled with a seed from the corpus, the rest will be filled one-by-one by the predictor.
		x = numpy.zeros([total_len + 1, seqlen, feature_vec_len])
		
		# Pick a random seed to kick the process off.
		start = numpy.random.randint(0, len(dataX) - init_len - 1)
		x[0:init_len] = dataX[start:start + init_len]
		#print("Seed ends with: >>%s<<\n" % ("".join(output_vec_to_char(dataY[start:start + init_len]))))

		generated_text = ""

		# Generate characters starting with that seed.
		for char_i in range(init_len, total_len):
			#print("Predicting char %d." % char_i)
			if stateful_lstm:
				# Initialize the LSTM
				model.reset_states()
				for batch_start in range(0, char_i, batch_size):
					model.predict_on_batch(x[batch_start:batch_start + batch_size])
				
				while True:
					# Predict a new character
					batch_start = (char_i // batch_size) * batch_size
					which_char_in_batch = char_i - batch_start
					
					output = model.predict_on_batch(x[batch_start:batch_start + batch_size])[which_char_in_batch]
					prediction = output_to_char(output) # , verbose=0

					generated_text += prediction

					x[char_i] = char_to_feature(prediction)
					
					#import pdb;pdb.set_trace()
					#with fullprint():
					#	print("Our corpus is: ", x)
					
					#sys.stderr.write('.')
					sys.stderr.write(prediction)
					sys.stderr.flush()
					
					if (which_char_in_batch == batch_size - 1):
						# We are at the end of the batch; let's continue with the next batch directly, without restarting.
						char_i += 1
						continue
					else:
						# We're not at the end of the batch. We have to restart in order to continue, because re-predicting on the same batch would screw the LSTM memory.
						break
			else:
				#with fullprint():
				#	print("Our corpus is: ", x)
				#print("Running prediction on batch %d-%d." % (char_i-batch_size, char_i - 1))
				output = model.predict_on_batch(x[char_i-batch_size:char_i])
				predictions = output_vec_to_char(output)
				#print("Predicted chars >>%s<<." % "".join(predictions))
				assert(len(predictions) == batch_size)
				prediction = predictions[-1]
				
				generated_text += prediction
				
				#print("Updating position %d with letter %s: features" % (char_i, prediction), char_to_feature(prediction))
				x[char_i, :seqlen-1] = x[char_i - 1, 1:]
				x[char_i, seqlen-1] = char_to_feature(prediction)
				
				#sys.stdout.write(prediction)
				#sys.stdout.flush()

		#print("Generated text: “%s”" % generated_text)
		#print("\n\n", generated_text)
		#print("\n--------------------------------------------------------------------------------\n\n")
		sys.stdout.write('%s\x00' % generated_text)
else:
	sys.stderr.write("Unrecognized action error.\n")
	sys.exit(1)
