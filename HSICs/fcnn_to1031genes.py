import pandas as pd
import numpy as np
import pickle
import gzip
import math


from numpy.random import seed
seed(1)
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

data = pd.read_csv("1031genedata.csv",index_col=0)
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
for i in range(0, 367, 74):
    if i == 296:
        train_index_list = list(set(range(0, 367)) - set(range(i, i + 74)))
        test_index_list = list(set(range(i, 367)))
        data_train = data.iloc[train_index_list]
        data_test = data.iloc[test_index_list]

    else:
        train_index_list = list(set(range(0, 367)) - set(range(i, i + 74)))
        test_index_list = list(set(range(i, i + 74)))
        data_train = data.iloc[train_index_list]
        data_test = data.iloc[test_index_list]
    y_train = y_values[train_index_list]
    y_test = y_values[test_index_list]

    # train_scaled = minmax_scale(data_train.as_matrix(), axis=0)
    # test_scaled = minmax_scale(data_test.as_matrix(), axis=0)

    scaler.fit(data_train.as_matrix())
    train_scaled = scaler.transform(data_train.as_matrix())
    test_scaled = scaler.transform(data_test.as_matrix())

    K1.set_session
    model = Sequential()
    model.add(Dense(650,
                    input_dim=train_scaled.shape[1], activation=tf.nn.relu, kernel_initializer='he_normal'))#tf.nn.relu
    model.add(Dropout(float(0.2)))
    model.add(Dense(256, activation=tf.nn.relu, kernel_initializer="he_normal"))
    #model.add(Dropout(float(0.2)))
    model.add(Dense(1, activation='linear', kernel_initializer="he_normal"))

    model.compile(loss='mean_squared_error', optimizer=K.optimizers.SGD(lr=0.00001, momentum=0.5))

    hist = model.fit(train_scaled, y_train, epochs=500, shuffle=True, batch_size=4,
                     validation_data=(test_scaled, y_test))
    y_pred = model.predict(test_scaled)
    np.savetxt("1031gen_NN_preds_fold_"+str(fold)+".csv", y_pred, delimiter=",")
    mse = mean_squared_error(y_test, y_pred)
    ort_mse += mse
    fold=fold+1
    K1.clear_session()


print("ortalama mse")
print(ort_mse/5)
print("fuck my life")

#51.315(dropout=0.2)