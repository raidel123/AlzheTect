#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# get main project path (in case this file is compiled alone)
if os.name == 'nt':
    # Windows
    context = os.getcwd().split('\\')
else:
    # Ubuntu
    context = os.getcwd().split('/')

context = '/'.join(context[:context.index('AlzheTect') + 1])

def CorrectResults(results_file):
    file_dp = pd.read_csv(results_file, low_memory=False)

    baselinedx = file_dp['BaselineDiagnosis']
    predictiondx = file_dp['Prediction']

    total = 0
    totalAD = 0
    totalCN = 0
    correct = 0
    correctAD = 0
    correctCN = 0

    for i in range(len(baselinedx)):
        if baselinedx[i] == predictiondx[i]:
            if baselinedx[i] == "AD":
                correctAD += 1
            else:
                correctCN += 1

            correct += 1

        if baselinedx[i] == "AD":
            totalAD += 1
        else:
            totalCN += 1

        total += 1

    print correctAD, '/', totalAD, '=', float(correctAD)/float(totalAD)
    print correctCN, '/', totalCN, '=', float(correctCN)/float(totalCN)

    total_result = ( float(correct)/float(total) ) * float(100)
    correctAD_result = ( float(correctAD)/float(totalAD) ) * float(100)
    correctCN_result = ( float(correctCN)/float(totalCN) ) * float(100)

    D = {'Total Correct': total_result, 'AD Correct': correctAD_result, 'CN Correct': correctCN_result}

    plt.ylim(ymax=100)
    plt.bar(range(len(D)), list(D.values()), tick_label=list(D.keys()), color=['r', 'g', 'b'], align='center')
    plt.savefig(context + '/trunk/results/ModelAccuracy.png')
    # plt.show()

    return  total_result, correctAD_result, correctCN_result
