import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import csv, os
import math
from PIL import Image
import copy
from keras.models import load_model
from tensorflow.python.keras.models import load_model
from lrp.LRPModel import LRPModel
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from keras.applications import *

class LRP:
    def __init__(self, model, state, prune_count, changed_neuron_index, pruned_neuron_location):
        self.model = model
        self.state = state
        self.prune_count = prune_count
        self.changed_neuron_index = changed_neuron_index
        self.pruned_neuron_location = pruned_neuron_location

    def build_model(self):
        inputs = keras.Input(shape=(100, 8, 1))
        flatten_output = layers.Flatten()(inputs)
        x = layers.Dense(800, activation='relu')(flatten_output)
        x1 = layers.Dense(300, activation='relu')(x)
        dense_output = layers.Dense(400, activation='linear')(x1)
        outputs = layers.Dense(4, activation='linear')(dense_output)

        model = keras.Model(inputs=inputs, outputs=outputs, name='prune_model')

        model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=0.001))

        return model

    def calculate_car_local(self, state):
        state_car_local = []
        state = np.reshape(state, (1, 800))
        for k in range(800):
            if state[0][k] == 0.5:
                state_car_local.append(k)
        return state_car_local

    def lrp_R(self):

        lrp_model = LRPModel(self.model)
        all_state_layer_R_list = []

        for i in range(len(self.state)):
            car_position = self.calculate_car_local(self.state[i])
            lrp_result, layer_R_list = lrp_model.perform_lrp(self.state[i])

            if len(all_state_layer_R_list) >= 2:
                for l in range(len(all_state_layer_R_list)):
                    all_state_layer_R_list[l] = np.sum([all_state_layer_R_list[l], layer_R_list[l]], axis=0)
            else:
                print("len(all_state_layer_R_list)-->",len(all_state_layer_R_list))
                print("len(layer_R_list)-->",len(layer_R_list))
                print("layer_R_list-->",layer_R_list)
                
                all_state_layer_R_list = np.sum([all_state_layer_R_list, layer_R_list], axis= -1)

        count = 0
        layer_R = np.zeros((1, 1504))
        for m in range(0, len(all_state_layer_R_list)):
            for n in range(len(all_state_layer_R_list[m])):
                layer_R[0][count] = all_state_layer_R_list[m][n]
                count += 1


        layer_R_list_reshape = layer_R.tolist()

        layer_R_list_reshape_new = layer_R_list_reshape[:]
        k = 0.01

        value_k = []
        index_k = []

        if len(self.changed_neuron_index) >= 0:
            for val in self.changed_neuron_index:
                layer_R_list_reshape_new[0][val] = float('inf')

        for j in range(int( k * len(layer_R_list_reshape_new[0]))):
            index_i = layer_R_list_reshape_new[0].index(min(layer_R_list_reshape_new[0]))
            value_k.append(layer_R_list_reshape_new[0][index_i])
            index_k.append(index_i)
            self.changed_neuron_index.append(index_i)
            layer_R_list_reshape_new[0][index_i] = float('inf')
        set0 = set(value_k)

        self.non_main_neuron_prune(index_k)
        print("==============================================")
        print("==============================================")


    def non_main_neuron_prune(self, index_k):

        model_weights = self.model.get_weights()

        new_weights = model_weights
        record_row = []
        record_col = []


        for j in index_k:
            
            if 0 <= j <= 799:
                row = 0
                record_row.append(row)
                record_col.append(j)
            if 800 <= j <= 1099:
                row = 2 
                record_row.append(row)
                col = int(j - 800)
                record_col.append(col)
            if 1100 <= j <= 1499:
                row = 4 
                record_row.append(row)
                col = int(j - 1100)
                record_col.append(col)
            if 1500 <= j <= 1503:
                row = 6
                record_row.append(row)
                col = int(j - 1500)
                record_col.append(col)

        pr = 0.1
        abs_target_weights = [[] for _ in range(len(record_row))]
        prune_index = [[] for _ in range(len(record_row))]

        which = 0
        for r,c in zip(record_row, record_col):
            for mm in range(len(model_weights[r])):
                abs_target_weights[which].append(abs(model_weights[r][mm][c]))
            which += 1

        which_ = 0
        for val in abs_target_weights:
            prune_num = int(len(val) * pr)
            for nn in range(prune_num):
                prune_index[which_].append(abs_target_weights[which_].index(min(abs_target_weights[which_])))
                abs_target_weights[which_][prune_index[which_][nn]] = float('inf')
            which_ += 1


        for aa in range(len(record_row)):
            self.pruned_neuron_location.append([record_row[aa], record_col[aa]])

        which__ = 0
        for m, n in zip(record_row, record_col):
            for r in prune_index[which__]:
                new_weights[m][r][n] = 0
            which__ += 1


        self.model.set_weights(new_weights)
        path = 'prune_model/model_' + str(self.prune_count)
        self.model.save(os.path.join(path, 'prune_model.h5'))


