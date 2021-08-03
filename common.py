'''import pandas as pd
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
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten,Conv1D, Dropout,AveragePooling1D,BatchNormalization,GlobalAveragePooling1D
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import keras
from keras.regularizers import l2
import time'''
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
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import np_utils
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


def define_rotulos(output,label_to_predict):
    """
    Obtém os rótulos finais diretamente do nome dos arquivos.
    """
    labels = []
    for i in range(len(output[label_to_predict])):
        lista = [output[label_to_predict][i]]*300
        labels.append(lista)
    flat_list = []
    for sublist in labels:
        for item in sublist:
            flat_list.append(item)

    labels_final = np.array(flat_list)
    return labels_final


def get_data_from_map(path):
    """
    Pega e separa os mapas simulados que serão utilizados no treimento e teste da rede.
    """
    f = open(path, 'r')
    lines = f.readlines()
    f.close()         
    total_values_angles = [line.split('\t')[1:] for line in lines]
    k = 0
    for array in total_values_angles:
        new_array = []
        for i in range(len(array)):
            if array[i] != '\n':
                new_array.append(float(array[i]))
        total_values_angles[k] = new_array
        k+=1
    
    
    specs = {}
    lab = 'a'
    n = 0
    data_training = []
    for i in range(301):
        specs[lab + str(i)] = []
        for j in range(len(total_values_angles)):
            specs[lab + str(i)].append(total_values_angles[j][n])
        n+=1
        data_training.append(specs[lab+str(i)])
    data_training = np.array(data_training)
    return data_training[1:]

def get_energies(path):
    """
    Pega o intervalo de energia no mapa.
    """
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    energies = []
    all_data = [line.split('\t') for line in lines]
    for line in all_data:
        if line[0] != '\n':
            energies.append(float(line[0]))
    
    return energies

def data_to_keras_format(X,y):
    """
    Normalização dos dados de entrada.
    """
    X = np.array(X).astype('float32')
    X = X.reshape(X.shape + (1,))
    X -= np.mean(X)
    X /= np.max(X)
    
    y = np.array(y)
    return X,y



def get_noise_train(X_train,y_train,size,nComp=10):
    """
    Insere ruído a partir de uma análise das componentes principais (PCA)
    nos dados de treinamento.
    """
    start = time.time()
    noise_aug = []
    noise = np.copy(X_train)
    mu = np.mean(noise, axis=0)
    pca = PCA()
    noise_model = pca.fit(noise)
    nComp = 10
    Xhat = np.dot(pca.transform(noise)[:,:nComp], pca.components_[:nComp,:])
    noise_level = np.dot(pca.transform(noise)[:,nComp:], pca.components_[nComp:,:])
    Xhat += mu
    SNR = np.linspace(1,5,size)
    for i in range(len(SNR)):
        noise_aug.append(SNR[i]*noise_level + Xhat)
        j = 0
        for spectra in noise_aug[i]:
            noise_aug[i][j] = spectra/np.max(spectra)
            print(f"Restam {len(noise_aug[i])-j} espectros!", end ='\r')
            j+=1
        print(f"Restam {len(SNR)-i}!", end ='\r')
    n_snr = len(noise_aug)
    m_noise = noise_aug[0].shape[0]
    X_train = np.array(noise_aug).reshape(n_snr*m_noise,689)
    y_train = [item for i in range(size) for item in y_train]
    print("Levou {:.2f} segundos para finalizar a adição de ruído.".format(time.time()-start))
    return X_train,y_train


def limit_map(n_slice,df,i=0):
    """
    Limita os intervalos do dataframe principal.
    """
    df_1 = pd.DataFrame(columns = df.columns)
    continua = True
    while continua:
        if i > len(df):
            continua = False
        temp = df.iloc[i:i+n_slice]
        df_1 = df_1.append(temp, ignore_index = True)
        i+=300
        #print("Restam {}: ".format(len(df)-i))
    df_1 = df_1.sample(frac=1).reset_index(drop=True)
    df_train = df_1.iloc[:,0:689]
    try:
        labels_final_1 = df_1['thickness'].tolist()
    except:
        labels_final_1 = []
    return df_train,labels_final_1


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize = (8,6))
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],label = 'Val Error')
    plt.legend()

    plt.figure(figsize = (8,6))
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['loss'],label='Train Error', linewidth = 2)
    plt.plot(hist['epoch'], hist['val_loss'],label = 'Val Error')
    plt.legend()
    plt.show()


def train_model(model, X_train,X_test, y_train,y_test, path_to_save,epochs=100,batch_size=300,seed=32):
    """
    Treina o modelo inserido e salva os melhores estados do treinamento.
    """
    np.random.seed(seed)
    best_model_file = path_to_save
    best_model = ModelCheckpoint(best_model_file, monitor='val_mae', verbose = 1, save_best_only = True)
    hist = model.fit(X_train,
                     y_train,
                     validation_data=(X_test, y_test),
                     epochs=epochs,
                     batch_size=batch_size,
                    callbacks = [best_model],
                     shuffle = True,
                     verbose=1,
                    use_multiprocessing=False)
    print("done")
    return hist
    
def validate_on_df(model, df_valid,n):
    
    df_validacao, y_validacao = limit_map(5,df_valid,i=n)
    X_validation, y_validacao = data_to_keras_format(df_validacao,y_validacao)
    X_validation -= np.mean(X_validation)
    X_validation /= np.max(X_validation)
    test_predictions = model.predict(X_validation)
    
    return test_predictions,y_validacao
    
    
class MEISmaps:

    def __init__(self, path):
        self.path = path
        self.map_files_names  = [f for f in listdir(path) if isfile(join(path,f))]
        self.output = {}
        self.labels = []
        self.labels_rotulo = ['dL', 'thickness', 'frac_m1', 'frac_m2']
        self.to_remove  = ['output_dL_','thickness_', '_frac_m1','_frac_m2','.dat']
        
    def get_rotulo_from_path(self,path):
        string_0 = path
        for pattern in self.to_remove:
            if pattern in string_0:
                string_0 = string_0.replace(pattern,'')
        values = string_0.split('_')    
        values = [float(v) for v in values]
        return values
    
    def get_data_from_map(self,path):
        """
        Pega e separa os mapas simulados que serão utilizados no treimento e teste da rede.
        """
        f = open(path, 'r')
        lines = f.readlines()
        f.close()         
        total_values_angles = [line.split('\t')[1:] for line in lines]
        k = 0
        for array in total_values_angles:
            new_array = []
            for i in range(len(array)):
                if array[i] != '\n':
                    new_array.append(float(array[i]))
            total_values_angles[k] = new_array
            k+=1
        specs = {}
        lab = 'a'
        n = 0
        data_training = []
        for i in range(301):
            specs[lab + str(i)] = []
            for j in range(len(total_values_angles)):
                specs[lab + str(i)].append(total_values_angles[j][n])
            n+=1
            data_training.append(specs[lab+str(i)])
        data_training = np.array(data_training)
        self.data_training = data_training
        return data_training[1:]
    
    def get_info_dict(self):
        paths = []
        output = {}
        labels = self.labels_rotulo
        for label in labels:
            output[label] = []
        k = 0
        for path_file in self.map_files_names:
            path_final = os.path.join(self.path, path_file)
            paths.append(path_final)
            values = self.get_rotulo_from_path(path_file)
            output['dL'].append(values[0])
            output['thickness'].append(values[1])
            output['frac_m1'].append(values[2])
            output['frac_m2'].append(values[3])
            k+=1
        self.output = output
        self.completed_paths = paths  
    
    
    def define_rotulos(self,label_to_predict='thickness'):
        """
        Obtém os rótulos finais diretamente do nome dos arquivos.
        """
        output = self.output
        labels = []
        for i in range(len(output[label_to_predict])):
            lista = [output[label_to_predict][i]]*300
            labels.append(lista)
        flat_list = []
        for sublist in labels:
            for item in sublist:
                flat_list.append(item)
    
        labels_final = np.array(flat_list)
        self.labels = labels_final
        return labels_final
    
    def prepare_dataFrame(self,lag = 12):
        # input coomo dict
        n_array = 689
        df = pd.DataFrame(columns = np.arange(0,n_array))
        start_time = time.time()
        k = 0
        datas = []
        lag = 12
        rotulos = []
        output = self.output
        paths = self.completed_paths
        for path in zip(paths):
            data = self.get_data_from_map(path[0])
            df1 = pd.DataFrame(data)
            values = []
            j = 0
            for i in range(int(df1.shape[0]/lag)):
                values.append(df1.iloc[j:j+lag].sum().tolist())
                rotulos.append(output['thickness'][k])
                j+=1

            df = df.append(values, ignore_index = True)
            k+=1
            datas.append(data)
            print("Restam {} mapas!".format(len(paths) - k),end = '\r')
        print("Tempo de execução de um mapa: {} s", (time.time() - start_time), end = '\r')
        df['thickness'] = rotulos
        self.df_completo  = df
        return df

    
    
    def get_dataframe(self):
        self.get_info_dict()
        df = self.prepare_dataFrame()
        
        return df