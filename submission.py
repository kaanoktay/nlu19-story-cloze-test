import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, os.path
import pickle
import tensorflow as tf
import sys
from sklearn.utils import shuffle
from Model import Model
import csv


# Network parameters
learning_rate   = 0.01
dim_embeddings  = 4800
training_epochs = 1000
batch_size      = 16
drop_rate       = 0.2

# The best model is obtained by using the last 3 story sentences with the ending
sent = "5432"
sample = "1"

X1 = tf.placeholder(tf.float32, [None, dim_embeddings])
X2 = tf.placeholder(tf.float32, [None, dim_embeddings])
X3 = tf.placeholder(tf.float32, [None, dim_embeddings])
X4 = tf.placeholder(tf.float32, [None, dim_embeddings])
X5 = tf.placeholder(tf.float32, [None, dim_embeddings])
y  = tf.placeholder(tf.float32, [None, 1])

w1 = tf.constant(value=0.0, dtype=tf.float32)
w2 = tf.constant(value=0.0, dtype=tf.float32)
w3 = tf.constant(value=0.0, dtype=tf.float32)
w4 = tf.constant(value=0.0, dtype=tf.float32)
w5 = tf.constant(value=0.0, dtype=tf.float32)

if("1" in sent):
    w1  = tf.Variable(initial_value=1,trainable= True, dtype=tf.float32 )
if("2" in sent):
    w2  = tf.Variable(initial_value=1,trainable= True, dtype=tf.float32 )
if("3" in sent):
    w3  = tf.Variable(initial_value=1,trainable= True, dtype=tf.float32 )
if("4" in sent):
    w4  = tf.Variable(initial_value=1,trainable= True, dtype=tf.float32 )
if("5" in sent):
    w5  = tf.Variable(initial_value=1,trainable= True, dtype=tf.float32 )

X  = w1*(X1+X5) + w2*(X2+X5) + w3*(X3+X5) + w4*(X4+X5) + w5*X5

# Hidden layer dimensions
n_hidden_1  = 2048    # 1st hidden layer
n_hidden_2  = 1028     # 2nd hidden layer
n_hidden_3  = 512     # 3rd hidden layer

# Initialization
initializer = tf.contrib.layers.xavier_initializer()

# MLP model
h1  = tf.layers.dense(X, n_hidden_1, activation=tf.nn.relu, kernel_initializer=initializer)
h1  = tf.nn.dropout(h1, rate=drop_rate*1.00)
h2  = tf.layers.dense(h1, n_hidden_2, activation=tf.nn.relu, kernel_initializer=initializer)
h2  = tf.nn.dropout(h2, rate=drop_rate*0.75)
h3  = tf.layers.dense(h2, n_hidden_3, activation=tf.nn.relu, kernel_initializer=initializer)
h3  = tf.nn.dropout(h3, rate=drop_rate*0.50)
prediction = tf.layers.dense(h3, 1, activation=tf.nn.sigmoid, kernel_initializer=initializer)

# Loss calculation and optimizer
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=prediction)
entropy_loss  = tf.reduce_mean(cross_entropy)
loss          = entropy_loss
optimizer     = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
saver         = tf.train.Saver()

filename = "VariableContribution/sent" + sent

model= Model()
X_train, Y_train, X_test_right, X_test_false, y_test_right, y_test_false = model.samples("Embeddings/embeddedVal.npy", permute=True, training_percent =90)
del model

n_train      = X_train.shape[0]
val_acc_hist = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(1,training_epochs+1):

        index= np.random.permutation(X_train.shape[0])
        X_train = X_train[index,:]
        Y_train = Y_train[index,:]

        if(epoch == 1):
            pred_val_right = sess.run(prediction, feed_dict={X1: X_test_right[:,0,:], X2: X_test_right[:,1,:],
                                                             X3: X_test_right[:,2,:], X4: X_test_right[:,3,:],
                                                             X5: X_test_right[:,4,:]})

            pred_val_false = sess.run(prediction, feed_dict={X1: X_test_false[:,0,:], X2: X_test_false[:,1,:],
                                                             X3: X_test_false[:,2,:], X4: X_test_false[:,3,:],
                                                             X5: X_test_false[:,4,:]})

            val_acc      = np.sum(pred_val_right>pred_val_false) / len(pred_val_right)
            max_acc      = val_acc
            val_acc_hist.append(val_acc)
            print("Session saved for epoch %d and val accuracy %f"%(epoch-1,val_acc))
            saver.save(sess, filename + sample + 'ckpt')

        for step in range(n_train//batch_size):

            first_idx = step*batch_size
            last_idx  = (step+1)*batch_size
            X_batch   =  X_train[first_idx:last_idx]
            Y_batch   =  Y_train[first_idx:last_idx]

            sess.run([loss, optimizer], feed_dict={X1: X_batch[:,0,:], X2: X_batch[:,1,:],
                                                   X3: X_batch[:,2,:], X4: X_batch[:,3,:],
                                                   X5: X_batch[:,4,:], y: Y_batch})



        pred_val_right = sess.run(prediction, feed_dict={X1: X_test_right[:,0,:], X2: X_test_right[:,1,:],
                                                         X3: X_test_right[:,2,:], X4: X_test_right[:,3,:],
                                                         X5: X_test_right[:,4,:]})

        pred_val_false = sess.run(prediction, feed_dict={X1: X_test_false[:,0,:], X2: X_test_false[:,1,:],
                                                         X3: X_test_false[:,2,:], X4: X_test_false[:,3,:],
                                                         X5: X_test_false[:,4,:]})

        val_acc      = np.sum(pred_val_right>pred_val_false) / len(pred_val_right)
        val_acc_hist.append(val_acc)
        if (val_acc > max_acc):
                print("Session saved for epoch %d and val accuracy %f"%(epoch,val_acc))
                saver.save(sess, filename + sample + 'ckpt')
                max_acc = val_acc
        if(epoch%20 == 0 ):
            print("Epoch: {:5}\tValidationAccuracy: {:.2%}".format(epoch, val_acc_hist[epoch]))
    np.save(filename + "_accuracy" + sample, np.array(val_acc_hist))

# Generating the submission file
model = Model()
X_test_ending1, X_test_ending2 = model.createTestSubmission("Embeddings/embeddedTestSubmission.npy")
del model

with tf.Session() as sess:
    saver.restore(sess,filename + sample +'ckpt')
    pred_test_ending1 = sess.run(prediction, feed_dict={X1: X_test_ending1[:,0,:], X2: X_test_ending1[:,1,:],
                                                        X3: X_test_ending1[:,2,:], X4: X_test_ending1[:,3,:],
                                                        X5: X_test_ending1[:,4,:]})

    pred_test_ending2 = sess.run(prediction, feed_dict={X1: X_test_ending1[:,0,:], X2: X_test_ending1[:,1,:],
                                                        X3: X_test_ending1[:,2,:], X4: X_test_ending1[:,3,:],
                                                        X5: X_test_ending1[:,4,:]})

    with open('SUBMISSION.csv', 'w') as csvFile:
        for j in range(pred_test_ending1.shape[0]):
            if pred_test_ending1[j] > pred_test_ending2[j]:
                writer = csv.writer(csvFile)
                writer.writerow("1")
            elif pred_test_ending2[j] > pred_test_ending1[j]:
                writer = csv.writer(csvFile)
                writer.writerow("2")
            else:
                a = (np.random.randint(2, size=1) + 1).item()
                writer = csv.writer(csvFile)
                writer.writerow(str(a))

    csvFile.close()
