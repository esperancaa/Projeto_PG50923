import pickle
import pandas as pd
import sys
import numpy as np
import os
import tensorflow 
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, LSTM, RNN, Bidirectional, Flatten, Activation, \
    RepeatVector, Permute, Multiply, Lambda, Concatenate, BatchNormalization
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import TimeDistributed, GRU
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import RMSprop, Adagrad
from transformers import TFBertModel
from tensorflow.keras.layers import SpatialDropout1D
from transformers import TFBertForSequenceClassification, BertConfig
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, recall_score, precision_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

seed_value = 42
#seed_value = None

np.random.seed(seed_value)
random.seed(seed_value)
tensorflow.random.set_seed(seed_value)

environment_name = sys.executable.split('/')[-3]
print('Environment:', environment_name)
os.environ[environment_name] = str(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow.compat.v1.keras.backend as K
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

select_gpus = [0]  
import os
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if select_gpus:
    devices = []
    for gpu in select_gpus:
        devices.append('/gpu:' + str(gpu))    
    strategy = tensorflow.distribute.MirroredStrategy(devices=devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

else:
    # Get the GPU device name.
    device_name = tensorflow.test.gpu_device_name()
    # The device name should look like the following:
    if device_name == '/device:GPU:0':
        print('Using GPU: {}'.format(device_name))
    else:
        raise SystemError('GPU device not found')

    os.environ["CUDA_VISIBLE_DEVICES"] = device_name
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

with open('KIBA.pkl', 'rb') as file:
    KIBA = pickle.load(file)

CHAR_PROT_SET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
				"U": 19, "T": 20, "W": 21,
				"V": 22, "Y": 23, "X": 24,
				"Z": 25 }

CHAR_SMILE_SET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, 
                 ".": 2, "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, 
                 "7": 38, "6": 6, "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64, '*':65}

PROT_LEN = 1198
SMILE_LEN = 176


def label_smiles(line, MAX_SMI_LEN, CHAR_SMILE_SET):
	X = np.zeros(MAX_SMI_LEN)
	for i, ch in enumerate(line[:MAX_SMI_LEN]): #	x, smi_ch_ind, y
		X[i] = CHAR_SMILE_SET[ch]

	return X #.tolist()

def label_sequence(line, MAX_SEQ_LEN, CHAR_PROT_SET):
	X = np.zeros(MAX_SEQ_LEN)

	for i, ch in enumerate(line[:MAX_SEQ_LEN]):
		X[i] = CHAR_PROT_SET[ch]

	return X #.tolist()


encoded_data = {'x_prot':[], 'x_met':[], 'y_train':[]}

for i, (index, row) in enumerate(KIBA.iterrows()):
     print('percentage: {0:.3%}'.format(i/KIBA.shape[0]), end='\r')
     encoded_data['x_prot'].append(label_sequence(row['target_sequence'], PROT_LEN, CHAR_PROT_SET))
     encoded_data['x_met'].append(label_smiles(row['compound_iso_smiles'], SMILE_LEN, CHAR_SMILE_SET))
     encoded_data['y_train'].append(row['ProteinID'])



smaller_dataset = {'x_prot': encoded_data['x_prot'][:220_000],
                     'x_met': encoded_data['x_met'][:220_000],
                     'y_train':encoded_data['y_train'][:220_000]}


x_prot = encoded_data['x_prot']
x_met = encoded_data['x_met']
y = encoded_data['y_train']


x_prot_train = np.array(x_prot[:200_000])
x_met_train = np.array(x_met[:200_000])
y_train = np.array(y[:200_000])

x_prot_val = np.array(x_prot[200_000:210_000])
x_met_val = np.array(x_met[200_000:210_000])
y_val = np.array(y[200_000:210_000])


x_prot_test = np.array(x_prot[210_000:])
x_met_test = np.array(x_met[210_000:])
y_test = np.array(y[210_000:])


def DeepDTA(PROT_LEN, SMILE_LEN, CHAR_SMI_SET_SIZE, CHAR_PROT_SET_SIZE, NUM_FILTERS):
    X_PROT = Input(shape=(PROT_LEN,), dtype='int32')
    X_MET = Input(shape=(SMILE_LEN,), dtype='int32') ### Buralar flagdan gelmeliii


    ### SMI_EMB_DINMS  FLAGS GELMELII 
    encode_protein = Embedding(input_dim=CHAR_PROT_SET_SIZE, output_dim=128, input_length=PROT_LEN)(X_PROT)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=4,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=8,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=12,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)
    
    encode_smiles = Embedding(input_dim=CHAR_SMI_SET_SIZE, output_dim=128, input_length=SMILE_LEN)(X_MET) 
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=4,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=6,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=8,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)


    encode_interaction = tf.keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) #merge.Add()([encode_smiles, encode_protein])

    # Fully connected 
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)


    # And add a logistic regression on top
    predictions = Dense(1, activation='sigmoid')(FC2) #kernel_initializer='normal"

    interactionModel = Model(inputs=[X_PROT, X_MET], outputs=[predictions])

    interactionModel.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    print(interactionModel.summary())
    return interactionModel

if select_gpus:
    with strategy.scope():
        model = DeepDTA(PROT_LEN, SMILE_LEN, len(CHAR_SMILE_SET), len(CHAR_PROT_SET), 32)

else:
    model = DeepDTA(PROT_LEN, SMILE_LEN, len(CHAR_SMILE_SET), len(CHAR_PROT_SET), 32)



keras_callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5),
       ModelCheckpoint('deepdta_checkpoint_clean_dataset.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
]

history = model.fit([x_prot_train, x_met_train], 
 						y_train,
 						epochs=50, 
 						batch_size=256, 
 						validation_data=([x_prot_val, x_met_val], y_val),
 						callbacks=keras_callbacks)
