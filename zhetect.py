#!/usr/bin/env python2

import sys
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Feature Engineering
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd() + "/utils")

import splitdata as sd
import dnnutils as dnnu
import parsingutils as pu

def TrainTestModel():
    train = pu.GetModelDataCSV(r"train/TADPOLE_D1.csv")
    test = pu.GetModelDataCSV(r"test/TADPOLE_D2.csv")

    '''
    print ('CN train data shape:', CN_train.shape)
    print ('AD train data shape:', AD_train.shape)

    print ('CN test data shape:', CN_test.shape)
    print ('AD test data shape:', AD_test.shape)
    '''

    train = train.drop(columns=['DX'])
    test = test.drop(columns=['DX'])

    # print (CN_train.head())
    # CN_train.to_cs("TADPOLE_D1_CN.csv",encoding='utf-8', index=False)

    train = pu.GenderToInt(train)
    test = pu.GenderToInt(test)

    print(train.dtypes)
    print(train.head())

    train = train.convert_objects(convert_numeric=True)

    print(train.dtypes)
    print(train.head())

    train = pu.nan_padding(train)
    train.to_csv("TADPOLE_D1_Train.csv",encoding='utf-8', index=False)

    train_x, train_y, valid_x, valid_y = pu.split_valid_test_data(train)
    print("train_x:{}".format(train_x.shape))
    print("train_y:{}".format(train_y.shape))
    print(train_x)
    print(train_y)

    print("valid_x:{}".format(valid_x.shape))
    print("valid_y:{}".format(valid_y.shape))

    model = dnnu.build_neural_network(train_x=train_x, train_y=train_y, valid_x=valid_x, valid_y=valid_y)

    epochs = 2000
    train_collect = 50
    train_print=train_collect*2

    learning_rate_value = 0.001
    batch_size=16

    x_collect = []
    train_loss_collect = []
    train_acc_collect = []
    valid_loss_collect = []
    valid_acc_collect = []

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        iteration=0
        for e in range(epochs):
            for batch_x,batch_y in dnnu.get_batch(train_x,train_y,batch_size):
                iteration+=1
                feed = {model.inputs: train_x,
                        model.labels: train_y,
                        model.learning_rate: learning_rate_value,
                        model.is_training:True
                       }

                train_loss, _, train_acc = sess.run([model.cost, model.optimizer, model.accuracy], feed_dict=feed)
                if iteration % train_collect == 0:
                    x_collect.append(e)
                    train_loss_collect.append(train_loss)
                    train_acc_collect.append(train_acc)

                    if iteration % train_print==0:
                         print("Epoch: {}/{}".format(e + 1, epochs),
                          "Train Loss: {:.4f}".format(train_loss),
                          "Train Acc: {:.4f}".format(train_acc))

                    feed = {model.inputs: valid_x,
                            model.labels: valid_y,
                            model.is_training:False
                           }
                    val_loss, val_acc = sess.run([model.cost, model.accuracy], feed_dict=feed)
                    valid_loss_collect.append(val_loss)
                    valid_acc_collect.append(val_acc)

                    if iteration % train_print==0:
                        print("Epoch: {}/{}".format(e + 1, epochs),
                          "Validation Loss: {:.4f}".format(val_loss),
                          "Validation Acc: {:.4f}".format(val_acc))


        saver.save(sess, "./alzheimer_detect.ckpt")

    plt.plot(x_collect, train_loss_collect, "r--")
    plt.plot(x_collect, valid_loss_collect, "g^")
    plt.show()Ap

    plt.plot(x_collect, train_acc_collect, "r--")
    plt.plot(x_collect, valid_acc_collect, "g^")
    plt.show()

    '''
    print ('CN train data shape:', train.shape)
    print ('AD train data shape:', test.shape)

    for val in train:
        print (train[val][0], type(train[val][0]))

    print ('CN train data shape:', train["PTGENDER"][0], type(train["PTGENDER"][0]))
    print ('AD train data shape:', train.head())
    '''

if __name__ == "__main__":
    TrainTestModel()
