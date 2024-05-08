from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os, keras
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8  
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

if __name__ == "__main__":
    for model_to_test in range(20, 0, -1):
        save_state = []
        save_action = []
        episode_seed = [10000, 9999, 10120, 11700, 8300, 9900, 25000, 65000]
        for i in range(8):
            import os
            from shutil import copyfile

            from testing_simulation import Simulation
            from generator import TrafficGenerator
            from model import TestModel
            from visualization import Visualization
            from utils import import_test_configuration, set_sumo, set_test_path

            config = import_test_configuration(config_file='testing_settings.ini')
            sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
            # model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'], episode_seed[i])
            # print("\n-------------model_{} is testing-------------\n".format(config['model_to_test']))
            model_path, plot_path = set_test_path(config['models_path_name'], model_to_test, episode_seed[i])
            print("\n-------------model_{} is testing-------------\n".format(model_to_test))
            print("model_path-->{}".format(model_path))
            print("plot_path-->{}".format(plot_path))

            Model = TestModel(
                model_path=model_path,
                input_dim_weight=config['num_states_width'],
                input_dim_height=config['num_states_height'],
            )

            TrafficGen = TrafficGenerator(
                config['max_steps'],
                config['n_cars_generated'],
            )

            Visualization = Visualization(
                plot_path,
                dpi=96
            )

            Simulation = Simulation(
                Model,
                TrafficGen,
                sumo_cmd,
                config['max_steps'],
                config['green_duration'],
                config['yellow_duration'],
                config['num_states_width'],
                config['num_states_height'],
                config['num_actions']
            )


            print('\n----- Test episode')
            # simulation_time = Simulation.run(episode_seed[i], save_state, save_action)
            simulation_time = Simulation.run(episode_seed[0], save_state, save_action)
            print('Simulation time:', simulation_time, 's')
            

            print("----- Testing info saved at:", plot_path)

            copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))

            with open(os.path.join(plot_path, 'total_negative_reward.txt'), "w") as file:
                file.write("%s\n" % np.sum(Simulation.reward_episode))
            Visualization.save_data_and_plot(data=Simulation.reward_episode, filename='reward', xlabel='Action step', ylabel='Reward')
            Visualization.save_data_and_plot(data=Simulation.queue_length_episode, filename='queue', xlabel='Step', ylabel='Queue length')
            Visualization.save_data_and_plot(data=Simulation.average_queue_length, filename='every-hour_queue_length', xlabel='Step', ylabel='Queue length')
            Visualization.save_data_and_plot(data=Simulation.average_wait_time, filename='every-hour_waiting_time', xlabel='hour', ylabel='Waiting time (vehicles)')
        np.save('save_data/state_for_lrp.npy', save_state)
        np.save('save_data/action_for_lrp.npy', save_action)