import tensorflow as tf
from tensorflow.contrib import learn
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn import datasets, linear_model
from sklearn import cross_validation
import numpy as np

import os
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
import pandas as pd
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import LabelEncoder
import sklearn

import random

np.random.seed(42)
tf.set_random_seed(42)

df_train = pd.DataFrame()
df_test = pd.DataFrame()
for file in glob("cnndata_125/*.csv"):
    if "train" in file:
        df_train = pd.concat([df_train,pd.read_csv(file, names = range(101), header=None)],ignore_index=True)
    else:
        df_test = pd.concat([df_test,pd.read_csv(file, names = range(101), header=None)],ignore_index=True)


## cheat 30 min
# df_train =  df_train[df_train[100] < 1800]
# df_test =  df_test[df_test[100] < 1800]

# print df_test


train_d = np.array(df_train)
test_d = np.array(df_test)


X_test = test_d[:,:-1]
Y_test = test_d[:,-1:]


total_len = train_d.shape[0]


# Parameters
learning_rate = 0.001
training_epochs = 50
batch_size = 100
display_step = 1
n_hidden = 256
n_input = X_test.shape[1]
n_classes = 1

# Create model
def multilayer_perceptron(x_mat, weights, biases):

    input_layer = tf.reshape(x_mat, [-1, 10, 10 , 1])
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[10, 5],
        padding="same",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 2], strides= (1,2))

    pool1_flat = tf.reshape(pool1, [-1, 10 * 5 * 16])

    dense = tf.layers.dense(inputs=pool1_flat, units=64, activation=tf.nn.relu)
    # Output layer with linear activation
    out_layer = tf.matmul(dense, weights['out']) + biases['out']
    return out_layer
# Store layers weight & bias

weights = {
    'out': tf.Variable(tf.random_normal([64, n_classes], 0, 0.1))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes], 0, 0.1))
}

# Construct model
# https://stackoverflow.com/questions/38399609/tensorflow-deep-neural-network-for-regression-always-predict-same-results-in-one
# pred = tf.transpose(multilayer_perceptron(x, weights, biases))
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, 1])
pred = multilayer_perceptron(x, weights, biases)
# Define loss and optimizer
cost = tf.reduce_mean(tf.square(pred-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

print ("Finish Initializing")
# Launch the graph

cost_list = []
cost_test = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Training cycle
    accuracy = sess.run(cost, feed_dict={x:X_test, y: Y_test})
    cost_test.append(accuracy)

    for epoch in range(training_epochs):
        np.random.shuffle(train_d)
        X_train = train_d[:,:-1]
        Y_train = train_d[:,-1:]
        avg_cost = 0.
        total_batch = int(total_len/batch_size)
        # Loop over all batches
        for i in range(total_batch-1):
            batch_x = X_train[i*batch_size:(i+1)*batch_size]
            batch_y = Y_train[i*batch_size:(i+1)*batch_size]

            # Run optimization op (backprop) and cost op (to get loss value)

            _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x,
                                                          y: batch_y})
            # print batch_y.shape
            # print p.shape
            # print c
            # exit()
            # Compute average loss
            avg_cost += c / total_batch

        # print (len(p))
        cost_list.append(avg_cost)
        # sample prediction
        label_value = batch_y
        # print (label_value)
        estimate = p
        # print estimate

        err = label_value - estimate

        accuracy = sess.run(cost, feed_dict={x:X_test, y: Y_test})
        # print accuracy
        # print X_test.shape[0]
        cost_test.append(accuracy)

        # print err
        print ("num batch:", total_batch)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
            print ("[*]----------------------------")
            for i in xrange(3):
                print ("label value:", label_value[i], \
                    "estimated value:", estimate[i])
            print ("[*]============================")

    print ("Optimization Finished!")

    # Test model
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    # accuracy = sess.run(cost, feed_dict={x:X_test, y: Y_test})
    predicted_vals = sess.run(pred, feed_dict={x: X_test})

    # print "accuracy"
    # print accuracy

    print sklearn.metrics.mean_absolute_error(Y_test, predicted_vals)
    print round(np.median(np.abs((predicted_vals - Y_test))), 2)
    print round(np.median(np.abs((predicted_vals - Y_test) / Y_test)) * 100, 2)
    #
    # print "predict_val"
    # print predicted_vals


    # print ("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))

pd.DataFrame(cost_list).to_csv("res.csv")
pd.DataFrame(cost_test).to_csv("res_test.csv")

# 116
# 357.508554523 218.09 13.26
# 266.839133151 180.88 14.0
# 157.132988157 111.73 15.51


# 125
# 131.365057679 89.53 9.09
# 90.1859049344 67.48 9.85


# 123
# 110.778063461 80.15 13.83
# 105.42245356 74.49 13.07
# 102.2348777 77.36 14.49
