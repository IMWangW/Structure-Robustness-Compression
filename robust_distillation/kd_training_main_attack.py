from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from kd_training_simulation_attack import Simulation
from generator import TrafficGenerator
from memory import Memory
from model import TrainModel
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path

import os, keras
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7  
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

if __name__ == "__main__":

    config = import_train_configuration(config_file='kd_training_settings_attack.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    
    S_model = TrainModel(
        config['num_layers'],
        config['width_layers'],
        config['width_layers_decrease'],
        config['batch_size'],
        config['learning_rate'],
        input_dim_weight=config['num_states_width'],
        input_dim_height=config['num_states_height'],
        output_dim=config['num_actions']
    )

    Memory = Memory(
        config['memory_size_max'],
        config['memory_size_min']
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'],
        config['n_cars_generated']
    )

    Simulation = Simulation(
        S_model,
        Memory,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states_width'],
        config['num_states_height'],
        config['num_actions'],
        config['training_epochs']
    )

    path = 'S_model/debug_hour_count_231029'
    Visualization = Visualization(
        path,
        dpi=96
    )

    episode = 0
    timestamp_start = datetime.datetime.now()


    while episode < config['total_episodes']:

        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])
        simulation_time, training_time = Simulation.run(episode, epsilon) 
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        episode += 1

        print("\n----- Start time:", timestamp_start)
        print("----- End time:", datetime.datetime.now())

    Visualization.save_data_and_plot(data=Simulation.negative_reward_store, filename='negative_reward', xlabel='Episode',
                                     ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=Simulation.positive_reward_store, filename='positive_reward', xlabel='Episode',
                                     ylabel='Cumulative positive reward')
