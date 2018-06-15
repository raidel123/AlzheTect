#!/usr/bin/env python

import sys
import os
import tensorflow as tf
import pandas as pd
import numpy as np

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

if __name__ == "__main__":

    # data = sd.SplitTrainTestData2()

    # Uncomment this block comment (below) to obtain results on the data

    #train = dnnu.GetModelDataCSV(context + r"/trunk/src/train/TADPOLE_train.csv")
    test = dnnu.GetModelDataCSV(context + r"/trunk/src/test/TADPOLE_test.csv")

    #dnnu.TrainModel(train)
    dnnu.TestModel(test)

    correct_prediction, correct_AD_prediction, correct_CN_prediction = pr.CorrectResults(context + r"/results/results.csv")
