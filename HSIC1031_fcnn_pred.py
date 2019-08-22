import pandas as pd
import numpy as np
import pickle
import gzip
import math
from tensorflow import set_random_seed
set_random_seed(2)
#import matplotlib.pyplot as plt
from keras import layers
#os.environ["CUDA_VISIBLE_DEVICES"]="3" #specify GPU
import keras as K
import tensorflow as tf
from keras import backend
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout,concatenate, Lambda
from keras.layers import merge  # works
from keras.engine.topology import Layer
#from keras.layers import Merge  # doesn't work
from keras.layers import merge
from keras.models import Model
from keras.optimizers import Adadelta
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras.models import model_from_yaml
from keras import backend as K1
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

data = pd.read_csv("1031andHSIC_train.csv",index_col=0)
print(data.shape)
print(data.columns.values)
data=shuffle(data)
y_values=data['GA'].values
print(y_values[0])

data=data.drop(['GA'], axis=1)
print(data.shape)

from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-10, 10))
fold=0
ort_mse=0

data_test = pd.read_csv("1031andHSIC_test.csv",index_col=0)
#test_scaled = scaler1.transform(data_test.values)
print(data_test.shape)

scaler.fit(data.as_matrix())
train_scaled = scaler.transform(data.as_matrix())
test_scaled = scaler.transform(data_test.as_matrix())

model = Sequential()
model.add(Dense(650,
                    input_dim=train_scaled.shape[1], activation=tf.nn.relu, kernel_initializer='he_normal'))#tf.nn.relu
model.add(Dropout(float(0.2)))
model.add(Dense(256, activation=tf.nn.relu, kernel_initializer="he_normal"))
#model.add(Dropout(float(0.2)))
model.add(Dense(1, activation='linear', kernel_initializer="he_normal"))

model.compile(loss='mean_squared_error', optimizer=K.optimizers.SGD(lr=0.00001, momentum=0.5))

hist = model.fit(train_scaled, y_values, epochs=500, shuffle=True, batch_size=4)
y_pred = model.predict(test_scaled)
np.savetxt("1031HSICgen_NN_preds.csv", y_pred, delimiter=",")