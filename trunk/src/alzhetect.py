#!/usr/bin/env python

import sys
import os
import tensorflow as tf
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

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

    train = dnnu.GetModelDataCSV(context + r"/trunk/src/train/TADPOLE_D1.csv")
    test = dnnu.GetModelDataCSV(context + r"/trunk/src/test/TADPOLE_D2.csv")

    # dnnu.TrainModel(train)
    dnnu.TestModel(test)

    # correct_prediction, correct_AD_prediction, correct_CN_prediction = pr.CorrectResults(context + r"/results/results.csv")
