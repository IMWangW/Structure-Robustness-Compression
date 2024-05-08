import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from keras import backend as K

class TrainModel:
    def __init__(self, num_layers, width,width_decrease,batch_size, learning_rate, input_dim_weight,input_dim_height, output_dim):
        self._input_dim_weight = input_dim_weight
        self._input_dim_height = input_dim_height
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width,width_decrease)


    def _build_model(self, num_layers, width,width_decrease):
        inputs = keras.Input(shape=(self._input_dim_weight, self._input_dim_height, 1))
        flatten_output = layers.Flatten()(inputs)
        x = layers.Dense(width, activation='relu')(flatten_output)

        x1 = layers.Dense(300, activation='relu')(x)

        dense_output = layers.Dense(400, activation='linear')(x1)
        outputs = layers.Dense(self._output_dim, activation='linear')(dense_output)

        model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
        model.summary()
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))
        return model


    def predict_one(self, state):
        state = np.reshape(state, [1,self._input_dim_weight,self._input_dim_height,1])
        return self._model.predict(state)


    def predict_batch(self, states):
        return self._model.predict(states)


    def train_batch(self, states, q_sa):
        self._model.fit(states, q_sa, epochs=1, verbose=0)


    def save_model(self, path):
        self._model.save(os.path.join(path, 'trained_model.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size




class TestModel:
    def __init__(self, model_path, input_dim_weight,input_dim_height):
        self._model = self._load_my_model(model_path)
        self._input_dim_weight = input_dim_weight
        self._input_dim_height = input_dim_height


    def _load_my_model(self, model_folder_path):
        """
        从指定的文件夹中加载存储的模型
        parameter:model_folder_path:加载模型的路径
        return：loaded_model：训练好的神经网络模型
        """
        # model_file_path = os.path.join(model_folder_path, 'ft_prune_model.h5')
        model_file_path = os.path.join(model_folder_path, 'attacked-ft_model.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            loaded_model.summary()
            return loaded_model
        else:
            sys.exit("Model number not found")


    def predict_one(self, state):
        """
        对于单个状态预测Q值
        """
        state = np.reshape(state, [1,self._input_dim_weight,self._input_dim_height,1])
        return self._model.predict(state)

