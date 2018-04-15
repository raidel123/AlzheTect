import tensorflow as tf
import numpy as np
import pandas as pd

# input data from tadpole mixed file and separate into training and testing

tadpole_dp = pd.read_csv(r"TADPOLE_D1_D2.csv", low_memory=False)

train_data = tadpole_dp.loc[tadpole_dp['D1'] == 1]
test_data = tadpole_dp.loc[tadpole_dp['D2'] == 1]

train_data.to_csv("TADPOLE_D1.csv", encoding='utf-8', index=False)
test_data.to_csv("TADPOLE_D2.csv", encoding='utf-8', index=False)

print ('Total data shape:', tadpole_dp.shape)
print ('Train data shape:', train_data.shape)
print ('Test data shape:', test_data.shape)
