import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-whitegrid')
import random
import os
import pickle
import re
from os import listdir
from os.path import isfile, join
import seaborn as sns
from sklearn.decomposition import PCA
#from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#from keras.utils import np_utils
#from tensorflow.keras.utils import np_utils
'''from keras.models import Sequential
from keras.layers import Dense, Flatten,Conv1D, Dropout,AveragePooling1D,BatchNormalization,GlobalAveragePooling1D
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import keras
from keras.regularizers import l2
import time
import sqlite3
from common import *
from keras.layers import Input, Dense
from keras.models import Model
#from keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras import initializers'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,Conv1D, Dropout,AveragePooling1D,BatchNormalization,GlobalAveragePooling1D
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#import keras
from tensorflow.keras.regularizers import l2
import time
import sqlite3
from common import *
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
#from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras import initializers


def conv_model(n_features):
    model = Sequential()
    activation = 'relu'
    model.add(Conv1D(9, 9, input_shape=(n_features,1), activation=activation))
    model.add(AveragePooling1D())
    model.add(BatchNormalization())
    model.add(Dropout(0.20))
    
    model.add(Conv1D(9, 7, activation=activation))
    model.add(AveragePooling1D())
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Conv1D(18, 7, activation=activation))
    model.add(AveragePooling1D())
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv1D(18, 5, activation=activation))
    model.add(AveragePooling1D())
    model.add(BatchNormalization())
    model.add(Dropout(0.35))
    
    model.add(Conv1D(36, 3, activation=activation,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                   bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5)))
    model.add(AveragePooling1D())
    model.add(BatchNormalization())
    model.add(Dropout(0.40))
    
    model.add(Conv1D(6, 1))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1))
    
    #optimizer = keras.optimizers.Adagrad(learning_rate = 0.001)
    optimizer = keras.optimizers.Adam(lr = 1e-3, decay = 1e-3/100)
    #optimizer = keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
    
    return model

class Architeture:
    
    def __init__(self, n_features):
        self.architetures = []
        self.performances = []
        self.insert_architeture(conv_model(n_features))
    
    def insert_architeture(self, arch):
        self.architetures.append(arch)
        
    def get_architetures(self):
        return self.architetures
        
    def insert_performance(self, perfo):
        self.performances.append(perfo)
    
    
