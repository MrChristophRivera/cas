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
decay_rate = [0.693]*62+[0.693/10]*63+[0.693/100]*62+[0.0]*63	# Create different decay rates for each unit

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

# Model
#
with tf.variable_scope('recurrent_layer_1'):
	cell = RWACell(num_cells, decay_rate=decay_rate)	# Modified code to run RWA model
	h, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

with tf.variable_scope('output_layer'):
	W_o = tf.get_variable('W_o', [num_cells, num_classes], initializer=tf.contrib.layers.xavier_initializer()),
	b_o = tf.get_variable('b_o', [num_classes], initializer=tf.constant_initializer(0.0)),
	h_last = h[:,num_steps-1,:]	# Grab values from the hidden state at the last step
	ly = tf.matmul(h_last, W_o)+b_o
	py = tf.nn.softmax(ly)

# Cost function and optimizer
#
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ly, labels=y))	# Cross-entropy cost function.
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Evaluate performance
#
correct = tf.equal(tf.argmax(py, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

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

		# Update parameters
		#
		cost_value, accuracy_value, _ = session.run((cost, accuracy, optimizer), feed_dict=feed_train)
		print('Iteration:', iteration, 'Cost:', cost_value/np.log(2.0), 'Accuracy:', 100.0*accuracy_value, '%')

	# Save the trained model
	#
	saver.save(session, 'bin/train.ckpt')

