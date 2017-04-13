#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2016-11-27
# Purpose: Train recurrent neural network to classify MNIST digits.
# License: See LICENSE
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import numpy as np
import tensorflow as tf
from RWACell import *	# Modified code to load RWACell

##########################################################################################
# Load data
##########################################################################################

from tensorflow.examples.tutorials.mnist import input_data
dataset = input_data.read_data_sets('MNIST_data/', one_hot=True)

##########################################################################################
# Settings
##########################################################################################

# Model settings
#
num_features = 1
num_steps = 28**2
num_cells = 250
num_classes = 10

# Training parameters
#
num_iterations = 250000
batch_size = 100
learning_rate = 0.001

##########################################################################################
# Operators
##########################################################################################

# Inputs
#
x = tf.placeholder(tf.float32, [batch_size, num_steps, num_features])
y = tf.placeholder(tf.float32, [batch_size, num_classes])

# Variables
#
decay = [0.693]*62+[0.0693]*63+[0.00693]*62+[0.0]*63	# Create different decay rates for different units
cell = RWACell(num_cells, decay=decay)	# Modified code to run RWA model
W_end = tf.Variable(tf.random_normal([num_cells, num_classes], stddev=np.sqrt(1.0/num_cells)))
b_end = tf.Variable(tf.zeros([num_classes]))

# Model
#
h, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
ly = tf.matmul(h[:,num_steps-1,:], W_end)+b_end
py = tf.nn.softmax(ly)

# Cost function and optimizer
#
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ly, labels=y))	# Cross-entropy cost function.
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Evaluate performance
#
correct = tf.equal(tf.argmax(py, 1), tf.argmax(y, 1))
accuracy = 100.0*tf.reduce_mean(tf.cast(correct, tf.float32))

# Create operator to initialize session
#
initializer = tf.global_variables_initializer()

# Create operator for saving the model and its parameters
#
saver = tf.train.Saver()

##########################################################################################
# Session
##########################################################################################

# Open session
#
with tf.Session() as session:

	# Initialize variables
	#
	session.run(initializer)

	# Each training session represents one batch
	#
	for iteration in range(num_iterations):

		# Grab a batch of training data
		#
		xs, ys = dataset.train.next_batch(batch_size)
		xs_sequence = np.reshape(xs, [batch_size, num_steps, num_features])	# Convert image into a sequence of pixels.
		feed_train = {x: xs_sequence, y: ys}

		# Periodically monitor progress
		#
		if iteration%100 == 0:

			# Grab a batch of validation data
			#
			xs, ys = dataset.validation.next_batch(batch_size)
			xs_sequence = np.reshape(xs, [batch_size, num_steps, num_features])	# Convert image into a sequence of pixels.
			feed_validation = {x: xs_sequence, y: ys}			

			# Print report to user
			#
			print('Iteration:', iteration)
			print('  Cost (Training):      ', cost.eval(feed_train)/np.log(2.0), 'bits')
			print('  Accuracy (Training):  ', accuracy.eval(feed_train), '%')
			print('  Cost (Validation):    ', cost.eval(feed_validation)/np.log(2.0), 'bits')
			print('  Accuracy (Validation):', accuracy.eval(feed_validation), '%')
			print('', flush=True)

		# Update parameters
		#
		session.run(optimizer, feed_dict=feed_train)

	# Save the trained model
	#
	saver.save(session, 'bin/train.ckpt')

