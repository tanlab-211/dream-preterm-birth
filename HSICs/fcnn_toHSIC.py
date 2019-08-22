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


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-10, 10))
fold=0
ort_mse=0


for i in range(0,5):
    train = pd.read_csv("HSIC_"+str(i)+"_train.csv", index_col=0)
    test=pd.read_csv("HSIC_"+str(i)+"_test.csv", index_col=0)
    y_train = train['GA'].values

    train = train.drop(['GA'], axis=1)
    print(train.shape)
    y_test = test['GA'].values

    test = test.drop(['GA'], axis=1)
    print(test.shape)

    scaler.fit(train.as_matrix())
    train_scaled = scaler.transform(train.as_matrix())
    test_scaled = scaler.transform(test.as_matrix())
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
    np.savetxt("1031gen_NN_preds_fold_" + str(fold) + ".csv", y_pred, delimiter=",")
    mse = mean_squared_error(y_test, y_pred)
    ort_mse += mse
    fold = fold + 1
    K1.clear_session()

print("ortalama mse")
print(ort_mse / 5)
print("fuck my life")
