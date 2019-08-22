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


bulbenisym=pd.read_csv("id_to_sym.csv")
print(bulbenisym.columns.values)
probes=list(bulbenisym['probes'])
genes=list(bulbenisym['genes'])

print(probes)

print(genes)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-10, 10))
num_rounds = 1000000
fold = 0
ort_mse = 0
asil_mse = 0
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
    data_HSIC_train = pd.read_csv("/home/isik/Desktop/dreamchallenge/HSICs/HSIC_" + str(fold) + "_train.csv")
    data_HSIC_test = pd.read_csv("/home/isik/Desktop/dreamchallenge/HSICs/HSIC_" + str(fold) + "_test.csv")
    HSIC_genisimleri = []
    for ii in data_HSIC_train.columns.values[0:len(data_HSIC_train.columns.values) - 1]:
        if ii.find('AFFX') == -1:
            inx = probes.index(ii)
            HSIC_genisimleri.append(genes[inx])
        else:
            HSIC_genisimleri.append(ii)

    print(HSIC_genisimleri)
    ortak = 0
    ortak_olmayan = []
    for ii1 in HSIC_genisimleri:
        if ii1 in data.columns.values:
            ortak = ortak + 1
            print(ii1)
        elif ii1.find('AFFX') == -1:
            inx = genes.index(ii1)
            ortak_olmayan.append(probes[inx])
        else:
            ortak_olmayan.append(ii1)

    print(ortak)
    print(len(ortak_olmayan))
    data_HSIC_train = data_HSIC_train[ortak_olmayan]
    data_HSIC_test = data_HSIC_test[ortak_olmayan]
    print(data_HSIC_train.shape)
    print(data_HSIC_test.shape)
    data_train = data_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)
    data_trains = pd.concat([data_train, data_HSIC_train], axis=1)
    data_tests = pd.concat([data_test, data_HSIC_test], axis=1)
    print(data_trains.shape)
    print(data_tests.shape)
    scaler.fit(data_trains.as_matrix())
    train_scaled = scaler.transform(data_trains.as_matrix())
    test_scaled = scaler.transform(data_tests.as_matrix())

    K1.set_session
    model = Sequential()
    model.add(Dense(650,
                    input_dim=train_scaled.shape[1], activation=tf.nn.relu,
                    kernel_initializer='he_normal'))  # tf.nn.relu
    model.add(Dropout(float(0.2)))
    model.add(Dense(256, activation=tf.nn.relu, kernel_initializer="he_normal"))
    # model.add(Dropout(float(0.2)))
    model.add(Dense(1, activation='linear', kernel_initializer="he_normal"))

    model.compile(loss='mean_squared_error', optimizer=K.optimizers.SGD(lr=0.00001, momentum=0.5))

    hist = model.fit(train_scaled, y_train, epochs=500, shuffle=True, batch_size=4,
                     validation_data=(test_scaled, y_test))
    y_pred = model.predict(test_scaled)
    np.savetxt("1031HSICgen_NN_preds_fold_" + str(fold) + ".csv", y_pred, delimiter=",")
    mse = mean_squared_error(y_test, y_pred)
    ort_mse += mse
    fold = fold + 1
    K1.clear_session()

print("ortalama mse")
print(ort_mse / 5)
print("fuck my life")