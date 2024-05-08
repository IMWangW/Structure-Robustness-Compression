from __future__ import absolute_import
from __future__ import print_function

# import tensorflow_model_optimization as tfmot
import numpy as np

import os
import datetime
from shutil import copyfile
import sys

import os, keras
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7  
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

if __name__ == "__main__":
        
    prune_episode = 20
    pruned_neuron_index = []
    pruned_neuron_location = []

    for i in range(1, prune_episode + 1):
        print("---------------prune_episode is {}---------------".format(i))
        from pruning_simulation_attack import Simulation
        from generator import TrafficGenerator
        from memory import Memory
        from utils import import_prune_configuration, set_sumo
        from LRP_neuron_pruner_attack import LRP
        import tensorflow as tf
        from tensorflow.keras import losses
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.models import load_model
        from visualization import Visualization
        from tensorflow.keras.utils import plot_model


        def _load_my_model(model_folder_path, model_folder):
            model_file_path = os.path.join(model_folder_path, model_folder)
            if model_folder_path == 'attacked-prune_mask-ft_model/model_0':
                loaded_model = load_model('Stu_model/model_2/trained_model.h5')
            else:
                if model_folder == 'attacked-prune_model.h5':
                    loaded_model = load_model(model_file_path)
                else:
                    loaded_model = load_model(model_file_path)

            return loaded_model

               
        state = np.load('save_data/state_for_lrp.npy')
        action = np.load('save_data/action_for_lrp.npy')

        new_state = np.arange(state.shape[0])
        np.random.shuffle(new_state)
        n = 20

        state_for_lrp = state[new_state[0:n]]

        config = import_prune_configuration(config_file='pruning_settings_attack.ini')
        sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
        # prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

        Memory = Memory(
            config['memory_size_max'],
            config['memory_size_min']
        )

        TrafficGen = TrafficGenerator(
            config['max_steps'],
            config['n_cars_generated']
        )

        path = 'attacked-prune_mask-ft_model/model_' + str(i)
        ori_model_folder = 'attacked-ft_model.h5'
        if i == 1:
            ori_model_folder = 'trained_model.h5'
            Model = _load_my_model('attacked-prune_mask-ft_model/model_' + str(i - 1), ori_model_folder)
            print("------------load kd model------------")
        else:
            Model = _load_my_model('attacked-prune_mask-ft_model/model_' + str(i - 1), ori_model_folder)

        lrp = LRP(Model, state_for_lrp, i, pruned_neuron_index, pruned_neuron_location)
        weights_mask = lrp.lrp_R()

        model_for_finetune = load_model('attacked-prune_mask-ft_model/model_' + str(i) + '/attacked-prune_model.h5')
        print("-----prune_model_{} is saved !-----".format(i))

        path = ('attacked-prune_mask-ft_model/model_' + str(i))
        Visualization = Visualization(
        path,
        dpi=96
        )


        Simulation = Simulation(
            model_for_finetune,
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
            config['training_epochs'],
            weights_mask
        )

        episode = 0
        timestamp_start = datetime.datetime.now()

        while episode < config['total_episodes']:
            print('\n----- Episode', str(episode + 1), 'of', str(config['total_episodes']))
            epsilon = 1.0 - (episode / config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
            simulation_time = Simulation.run(episode, epsilon, pruned_neuron_location)  # run the simulation
            print('Simulation time:', simulation_time, 's ', ' - Total:',round(simulation_time , 1), 's')
            episode += 1

        save_ft_model_path = 'attacked-prune_mask-ft_model/model_' + str(i)
        model_for_finetune.save(os.path.join(save_ft_model_path, 'attacked-ft_model.h5'))
        try:
            with open(os.path.join(save_ft_model_path, 'ft_total_negative_reward.txt'), "w") as file:
                file.write("%s\n" % Simulation._sum_neg_reward)
        except Exception as e:
            print("++++++++++======= neg_reward error was happened:{} =======++++++++++".format(e))
        try:
            plot_model(model_for_finetune, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)
        except Exception as e:
            print("++++++++++======= plot_model error was happened:{} =======++++++++++".format(e))
        
        Visualization.save_data_and_plot(data=Simulation.negative_reward_store, filename='ft_negative_reward', xlabel='Episode', ylabel='Cumulative negative reward')

        print("\n----- Start time:", timestamp_start)
        print("----- End time:", datetime.datetime.now())
        print("----- Session info saved at:", save_ft_model_path)

        copyfile(src='pruning_settings.ini', dst=os.path.join(path, 'pruning_settings.ini'))
