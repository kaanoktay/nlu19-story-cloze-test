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

# For each model
for sent in ["54321", "5432", "543", "54", "5"]:

    filename = "VariableContribution/sent" + sent

    # Input construction for MLP network
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
    n_hidden_1  = 2048   # 1st hidden layer
    n_hidden_2  = 1024   # 2nd hidden layer
    n_hidden_3  = 512    # 3rd hidden layer

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

    # For 5 different random samples
    for sample in range(1, 6):
        print("\nSample " + str(sample) + " of Model " + sent + ":\n")

        model  =  Model()
        X_train, Y_train, X_test_right, X_test_false, y_test_right, y_test_false = model.samples("Embeddings/embeddedVal.npy", permute=True)
        del model

        n_train      = X_train.shape[0]
        val_acc_hist = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # For each epoch
            for epoch in range(1, training_epochs+1):
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
                    saver.save(sess, filename + str(sample) + 'ckpt')

                # Take mini batches
                for step in range(n_train//batch_size):

                    first_idx = step*batch_size
                    last_idx  = (step+1)*batch_size
                    X_batch   = X_train[first_idx:last_idx]
                    Y_batch   = Y_train[first_idx:last_idx]

                    _, _ = sess.run([loss, optimizer], feed_dict={X1: X_batch[:,0,:], X2: X_batch[:,1,:],
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

                # Save the graph if val accuracy increases
                if (val_acc > max_acc):
                    print("Session saved for epoch %d and val accuracy %f"%(epoch,val_acc))
                    saver.save(sess, filename + str(sample) + 'ckpt')
                    max_acc = val_acc

                if(epoch%20== 0 ):
                    print("Epoch: {:5}\tValidationAccuracy: {:.2%}".format(epoch, val_acc_hist[epoch]))

            np.save(filename + "_accuracy" + str(sample), np.array(val_acc_hist))

        model  =  Model()
        _, _, x_test_right, x_test_false, _, _ = model.samples("Embeddings/embeddedTestReport.npy", permute=True, training_percent=0)
        del model


        with tf.Session() as sess:
            saver.restore(sess, filename + str(sample) + 'ckpt')

            pred_test_right = sess.run(prediction, feed_dict={X1: x_test_right[:,0,:], X2: x_test_right[:,1,:],
                                                              X3: x_test_right[:,2,:], X4: x_test_right[:,3,:],
                                                              X5: x_test_right[:,4,:]})

            pred_test_false = sess.run(prediction, feed_dict={X1: x_test_false[:,0,:], X2: x_test_false[:,1,:],
                                                              X3: x_test_false[:,2,:], X4: x_test_false[:,3,:],
                                                              X5: x_test_false[:,4,:]})

            test_acc = np.sum(pred_test_right>pred_test_false) / len(pred_test_right)
            print("Accuracy on test set", test_acc)
            np.save(filename + "_Testset_accuracy" + str(sample), np.array(test_acc))


# Find the best model
max_accuracy = 0
for sent in ["54321", "5432", "543", "54", "5"]:
    filename = "VariableContribution/sent" + sent + "_accuracy"
    accuracy = np.expand_dims(np.load(filename+"1"+".npy"), axis=1)

    for i in range(2,6):
        accuracy= np.concatenate((accuracy, np.expand_dims(np.load(filename + str(i) + ".npy"), axis=1)), axis=1)

    val_avg_acc = np.mean(np.max(accuracy, axis=0))

    if val_avg_acc > max_accuracy:
        best_model   = sent
        max_accuracy = val_avg_acc

filename = "VariableContribution/sent" + best_model + "_accuracy"
accuracy = np.expand_dims(np.load(filename+"1"+".npy"), axis=1)
for i in range(2,6):
    accuracy = np.concatenate((accuracy, np.expand_dims(np.load(filename+str(i)+".npy"), axis=1)), axis=1)

best_sample = str(np.argmax(np.max(accuracy, axis=0))+1)

print("Best MLP network is the sample " + best_sample + " of " + best_model)

label =[]
plt.figure(figsize=[10,5])
for (sent,color) in [(54321,'red'), (5432, 'green'), (543,'blue'), (54, 'black'), (5, 'pink')]:
    filename = "VariableContribution/sent"+str(sent)+"_accuracy"
    acc = np.expand_dims(np.load(filename+"1"+".npy"), axis=1)
    for i in range(2,6):
        acc= np.concatenate((acc, np.expand_dims(np.load(filename+str(i)+".npy"), axis=1)), axis=1)

    l, = plt.plot(np.mean(acc, axis=1), color = color)
    label.append(l)

    plt.ylim([0.4,.8])
    print("Sent "+str(sent)+"; Max accuracy in average on Validation data: "+str(np.mean(np.max(acc, axis=0))))

plt.ylabel("Accuracy", fontsize=12)
plt.xlabel("Epochs", fontsize=12)
plt.legend(label, [ "54321","5432", "543", "54", "5"] )
plt.title("Average validation accuracies vs epoch")
plt.show()

label = []
plt.figure(figsize=[10,5])
for (sent,color) in [(54321,'red'), (5432, 'green'), (543,'blue'), (54, 'black'), (5, 'pink')]:
    filename = "VariableContribution/sent"+str(sent)+"_Testset_accuracy"
    acc = np.load(filename+"1"+".npy")
    for i in range(2,6):
        acc= np.append(acc,np.load(filename+str(i)+".npy"))

    l,=plt.plot(np.arange(1,6), acc, color = color )
    label.append(l)
    plt.ylim([0.6, .8])
    print("Sent "+str(sent)+"; Accuracy in average on Test data:"+str(np.mean(acc)))
plt.ylabel("Accuracy", fontsize=12)
plt.xlabel("Experiments", fontsize=12)
plt.title("Average test accuracies vs experiment iterations")
plt.xticks(np.arange(1,6))
plt.xlim([0.75, 5.25])
plt.legend(label ,[ "54321","5432", "543", "54", "5"] )
plt.show()
