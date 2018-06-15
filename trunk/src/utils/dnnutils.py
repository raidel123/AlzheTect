#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import sys
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Build Neural Network
from collections import namedtuple

# Feature Engineering
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# get main project path (in case this file is compiled alone)
if os.name == 'nt':
    # Windows
    context = os.getcwd().split('\\')
else:
    # Ubuntu
    context = os.getcwd().split('/')

context = '/'.join(context[:context.index('AlzheTect') + 1])

sys.path.append(context + "/trunk/src/utils")

import splitdata as sd
import dnnutils as dnnu
import plotresults as pr

def build_neural_network(data_x, hidden_units=10):
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[None, data_x.shape[1]])
    labels = tf.placeholder(tf.float32, shape=[None, 1])
    learning_rate = tf.placeholder(tf.float32)
    is_training=tf.Variable(True,dtype=tf.bool)

    initializer = tf.contrib.layers.xavier_initializer()
    fc = tf.layers.dense(inputs, hidden_units, activation=None,kernel_initializer=initializer)
    fc=tf.layers.batch_normalization(fc, training=is_training)
    fc=tf.nn.relu(fc)

    logits = tf.layers.dense(fc, 1, activation=None)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    cost = tf.reduce_mean(cross_entropy)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    predicted = tf.nn.sigmoid(logits)
    correct_pred = tf.equal(tf.round(predicted), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Export the nodes
    export_nodes = ['inputs', 'labels', 'learning_rate','is_training', 'logits',
                    'cost', 'optimizer', 'predicted', 'accuracy']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph

def get_batch(data_x,data_y,batch_size=32):
    batch_n=len(data_x)//batch_size
    for i in range(batch_n):
        batch_x=data_x[i*batch_size:(i+1)*batch_size]
        batch_y=data_y[i*batch_size:(i+1)*batch_size]

        yield batch_x,batch_y

def TestModel(test):

    test_diagnosis_id = test["RID"]
    baseline_diagnosis = test["DX_bl"]

    test = test.drop(columns=['DX_bl'])
    test = test.drop(columns=['DX'])
    test = test.drop(columns=['RID'])
    test = test.drop(columns=['EXAMDATE'])
    test = TransformData(test)

    # test_x, test_y, test_x, test_y = get_valid_test_data(train)

    model = dnnu.build_neural_network(data_x=test)
    restorer = tf.train.Saver()
    with tf.Session() as sess:
        restorer.restore(sess,context + r"/trunk/src/trained_model/alzheimer_detect.ckpt")
        feed={
            model.inputs:test,
            model.is_training:False
        }
        test_predict=sess.run(model.predicted,feed_dict=feed)

    # print test_predict

    print test_predict
    binarizer=Binarizer(0.5)
    test_predict_result=binarizer.fit_transform(test_predict)
    test_predict_result=test_predict_result.astype(np.int32)
    # print test_predict_result[:10]

    diagnosis_id = test_diagnosis_id.copy()
    evaluation = diagnosis_id.to_frame()
    evaluation["BaselineDiagnosis"] = baseline_diagnosis
    evaluation["Prediction"] = [ ResulUnbinarizer(val) for val in test_predict_result ]
    # print evaluation[:10]

    evaluation.to_csv(context + r"/results/results.csv",index=False)

def TrainModel(train):

    train = train.drop(columns=['DX'])
    train = train.drop(columns=['RID'])
    train = train.drop(columns=['EXAMDATE'])
    train = TransformData(train)

    train_x, train_y, valid_x, valid_y = get_valid_test_data(train)
    print("train_x:{}".format(train_x.shape))
    print("train_y:{}".format(train_y.shape))
    print(train_x)
    print(train_y)

    print("valid_x:{}".format(valid_x.shape))
    print("valid_y:{}".format(valid_y.shape))

    model = dnnu.build_neural_network(data_x=train_x)

    epochs = 2000
    train_collect = 50
    train_print=train_collect*2

    learning_rate_value = 0.001
    batch_size=12

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


        saver.save(sess, context + r"/trunk/src/trained_model/alzheimer_detect.ckpt")

    plt.plot(x_collect, train_loss_collect, "r--")
    plt.plot(x_collect, valid_loss_collect, "g^")
    plt.show()

    plt.plot(x_collect, train_acc_collect, "r--")
    plt.plot(x_collect, valid_acc_collect, "g^")
    plt.show()

def TransformData(data):
    # TODO: remove print
    print ('Data train data shape:', data.shape)
    transformed = GenderToInt(data)
    transformed = transformed.convert_objects(convert_numeric=True)
    transformed = nan_padding(transformed)

    return transformed

def ResulUnbinarizer(val):
    if val == 0:
        return "AD"
    else:
        return "CN"

# if any of the fields in the top line are changed in the list below
# change the value within nan_padding() function for the relevant fields
# starting positions
def GetRelevantFields():
    return  [
            'RID','DX_bl','DX','EXAMDATE',
            'MMSE', 'MMSE_bl', 'ADAS11', 'ADAS13','CDRSB', 'RAVLT_immediate',
            'Hippocampus','WholeBrain','Entorhinal', 'MidTemp',
            'FDG','AV45',
            'ABETA_UPENNBIOMK9_04_19_17','TAU_UPENNBIOMK9_04_19_17','PTAU_UPENNBIOMK9_04_19_17',
            'APOE4','AGE','PTGENDER',
            'ST83TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST83TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST83CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
            'ST24TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST24TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST24CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST113CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST54CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST39TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16','ST39TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
            'ST39SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST88SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','RIGHT_HIPPOCAMPUS_UCBERKELEYAV45_10_17_16','RIGHT_HIPPOCAMPUS_SIZE_UCBERKELEYAV45_10_17_16','RIGHT_HIPPOCAMPUS_UCBERKELEYAV1451_10_17_16','RIGHT_HIPPOCAMPUS_SIZE_UCBERKELEYAV1451_10_17_16',
            'ST29SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','LEFT_HIPPOCAMPUS_UCBERKELEYAV45_10_17_16','LEFT_HIPPOCAMPUS_SIZE_UCBERKELEYAV45_10_17_16','LEFT_HIPPOCAMPUS_UCBERKELEYAV1451_10_17_16','LEFT_HIPPOCAMPUS_SIZE_UCBERKELEYAV1451_10_17_16',
            'ST89SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST30SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST130TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST130TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST82TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16','ST82TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
            'ST12SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','LEFT_AMYGDALA_UCBERKELEYAV45_10_17_16','LEFT_AMYGDALA_SIZE_UCBERKELEYAV45_10_17_16','LEFT_AMYGDALA_UCBERKELEYAV1451_10_17_16','LEFT_AMYGDALA_SIZE_UCBERKELEYAV1451_10_17_16',
            'ST40TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST40TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST117SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST109TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16','ST109TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16','ST109TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST109TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST110TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST110TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            ]

# getting cannot convert string to float errors
def nan_padding(data):
    for column in GetRelevantFields()[4:]:
        imputer=Imputer()
        data[column]=imputer.fit_transform(data[column].values.reshape(-1,1))
    return data

def GenderToInt(data):
    le = LabelEncoder()
    le.fit(["Male","Female"])
    data["PTGENDER"]=le.transform(data["PTGENDER"])
    return data

def GetModelDataCSV(indata):
    # load data (low memory: supress memory warning)
    file_dp = pd.read_csv(indata, low_memory=False)

    model_dp = file_dp[GetRelevantFields()]

    # return model_dp
    return sd.SplitClassData(indata=model_dp, file=False)

def GetClass(data):
    fields = []
    for val in data:
        if val == "AD":
            fields.append(np.array([np.int64(0)]))
        elif val == 'CN':
            fields.append(np.array([np.int64(1)]))
        else:
            fields.append(np.array([np.int64(2)]))

    fields = np.array(fields)
    return fields

def get_valid_test_data(data, fraction=(1 - 0.8)):
    data_y = data["DX_bl"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)
    #data_y = GetClass(data_y)
    print 'type:', type(data_y), data_y
    print 'innertype:', type(data_y[0]), data_y[0]
    print 'innerinnertype:', type(data_y[0][0]), data_y[0][0]
    # print 'type', type(data_y)

    data_x = data.drop(["DX_bl"], axis=1)

    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction)

    return train_x.values, train_y, valid_x, valid_y
