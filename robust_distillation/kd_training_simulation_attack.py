import traci
import numpy as np
import random
import timeit
import tensorflow as tf
import os
# import tensorflow.compat.v1 as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from keras import backend as K
from trattacker import Trogantraining
# import keras
from tensorflow.keras import losses
import copy
from tensorflow.keras.optimizers import Adam


# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration,
                 num_states_width, num_states_height, num_actions, training_epochs):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states_width = num_states_width
        self._num_states_height = num_states_height
        self._num_states = (num_states_width,num_states_height)
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._queue_length_episode = []
        self._avg_queue_length_store = []
        self.ori_loss_store = []
        self.kl_loss_store = []
        self.hint_loss_store = []
        self.total_loss_store = []
        self.negative_reward_store = []
        self.positive_reward_store = []
        self._training_epochs = training_epochs
        
        self.average_wait_time = [[] for _ in range(24)]        
        self.leave_car_id = [[] for _ in range(24)]
        self.hour_wait_time = [[] for _ in range(24)]
        self.new_all_wait_time = 0
        
        self.sum_positive_reward = 0



    def run(self, episode, epsilon):
       
        start_time = timeit.default_timer()
        self.episode = episode 

        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

              
        self._step = 0
        hour_count = 1
        self._queue_length_episode = []
        self._waiting_times = {}
        self.new_waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1

        T_model_folder_path = 'T_model'
        T_model_file_path = os.path.join(T_model_folder_path, 'trained_model.h5')
        self.T_model = load_model(T_model_file_path)

        self.T_model.summary()

        if episode == 0:
            S_model_folder_path = 'S_model/exploration_data_KD_model/model_1'
            S_model_file_path = os.path.join(S_model_folder_path, 'trained_model.h5')
            self.S_model = load_model(S_model_file_path)
        else:
            S_model_folder_path = 'S_model/debug_hour_count_231029/model_' + str(episode)
            S_model_file_path = os.path.join(S_model_folder_path, 'trained_model.h5')
            self.S_model = load_model(S_model_file_path)

        while self._step < self._max_steps:
            current_state = self._get_state()
            queue_length = self._get_queue_length()
            self._queue_length_episode.append(queue_length)
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait


            if self._step != 0:
                attack = Trogantraining(old_state, old_action, reward, epsilon)
                old_state, old_action, reward = attack.train_attack_main()
                self._Memory.add_sample((old_state, old_action, reward, current_state))
         
            action = self._choose_action(current_state, epsilon)

                        
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            self._set_green_phase(action)
            self._simulate(self._green_duration)

            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            if reward < 0:
                self._sum_neg_reward += reward
            if reward > 0:
                self.sum_positive_reward += reward

            if self._step>(3600*(hour_count+3)):
                for i in range(3):
                    print("Training...")
                    self.train_step()
                hour_count += 1



        print("hour_count-->",hour_count+3)
        print("-- Total negative reward: {}\t -- Total positive reward:{}\t".format(self._sum_neg_reward, self.sum_positive_reward), "-- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)


        start_time = timeit.default_timer()
        path = 'S_model/debug_hour_count_231029/model_' + str(episode + 1)
        self.S_model.save(os.path.join(path, 'trained_model.h5'))
        with open(os.path.join(path, 'attack-train_total_negative_reward.txt'), "w") as file:
            file.write("%s\n" % self._sum_neg_reward)
        print("----- Session info saved at:", path)
        
        self.negative_reward_store.append(self._sum_neg_reward)
        print("every-epoch cumulative negative reward-->{}".format(self.negative_reward_store))
        self.positive_reward_store.append(self.sum_positive_reward)
        print("every-epoch cumulative positive reward-->{}".format(self.positive_reward_store))
        
        from visualization import Visualization
        Visualization = Visualization(
            path,
            dpi=96
        )
        Visualization.save_data_and_plot(data=self._queue_length_episode, filename='queue_length', xlabel='training_epochs', ylabel='queue_length')
        try:
            self.every_hour_wait_times()
            Visualization.save_data_and_plot(data=self.average_wait_time, filename='every-hour_waiting_time', xlabel='hour', ylabel='Waiting time (vehicles)')
        except Exception as e:
            print("++++++++++======= error was happened:{} =======++++++++++".format(e))
        Visualization.save_data_and_plot(data=self.ori_loss_store, filename='ori_loss', xlabel='training_epochs', ylabel='ori_loss')
        Visualization.save_data_and_plot(data=self.kl_loss_store, filename='kl_loss', xlabel='training_epochs', ylabel='kl_loss')
        Visualization.save_data_and_plot(data=self.hint_loss_store, filename='hint_loss', xlabel='training_epochs', ylabel='hint_loss')
        Visualization.save_data_and_plot(data=self.total_loss_store, filename='total_loss', xlabel='training_epochs',ylabel='total_loss')
        training_time = round(timeit.default_timer() - start_time, 1)


        return simulation_time, training_time


    def _simulate(self, steps_todo):
        
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()              
            self._step += 1            
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length
            


    def _collect_waiting_times(self):
        
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  
            if road_id in incoming_roads:  
                self._waiting_times[car_id] = wait_time
                self.new_waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: 
                    del self._waiting_times[car_id]
                    for i in range(24):
                        if i*3600 <= self._step < (i+1)*3600:
                            self.leave_car_id[i].append(car_id)
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def every_hour_wait_times(self):
        every_hour_wait_time = [[] for _ in range(24)]
        every_hour_leave_car_number = []
        average_wait_time = [[] for _ in range(24)]
        for i in range(len(self.leave_car_id)):
            every_hour_leave_car_number.append(len(self.leave_car_id[i]))
        for i in range(len(self.leave_car_id)):
            for j in range(len(self.leave_car_id[i])):
                self.hour_wait_time[i].append(self.new_waiting_times.get(self.leave_car_id[i][j]))
        for i in range(len(self.hour_wait_time)):
            wait_time = 0
            for j in range(len(self.hour_wait_time[i])):
                wait_time += self.hour_wait_time[i][j]
            every_hour_wait_time[i].append(wait_time)
        for i in range(24):
            if every_hour_leave_car_number[i] != 0:
                self.average_wait_time[i].append(every_hour_wait_time[i][0]/every_hour_leave_car_number[i])
            else:
                self.average_wait_time[i].append(0)
        self.average_wait_time = [i for j in self.average_wait_time for i in j]


    def _choose_action(self, state, epsilon):
        return np.argmax(self.S_model.predict(np.reshape(state, (1, 100, 8, 1))))  
                
    def _set_yellow_phase(self, old_action):
        
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase_code)


    def _set_green_phase(self, action_number):
       
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)


    def _get_queue_length(self):
       
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length


    def _get_state(self):
        global car_position
        state = np.zeros(self._num_states_width*self._num_states_height)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 700 - lane_pos
            lane_cell = int(lane_pos / 7)
            if lane_pos % 7 == 0 and lane_pos != 0:
                lane_cell -= 1

            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 7:
                if lane_cell >= 0 and lane_cell <= 9:
                    car_position = int(str(lane_group * 10) + str(lane_cell))                  
                else:
                    car_position = int(str(lane_group) + str(lane_cell))
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False 

            if valid_car:
                state[car_position] = 1
        state = state.reshape(self._num_states_width,self._num_states_height)
        state = state - 0.5

        return state


    def train_step(self):
        print("memory size is-->{}".format(self._Memory.size_now()))
        batch = self._Memory.get_samples(self._Memory.size_now())
        print("len(batch)-->{}".format(len(batch)))
        if len(batch) > 0:            
            states = np.array([val[0] for val in batch])
            next_states = np.array([val[3] for val in batch])
            temp = 10
            ori_weight = [0.0001, 0.00013492828476735632, 0.000182056420302608, 0.00024564560522315806, 0.00033144540173399864, 0.00044721359549995785, 0.0006034176336545161, 0.0008141810630738084, 0.0010985605433061173, 0.0014822688982138947, 
                            0.0019999999999999987, 0.002698565695347125, 0.003641128406052158, 0.004912912104463158, 0.006628908034679968, 0.00894427190999915, 0.012068352673090315, 0.01628362126147616, 0.02197121086612233, 0.029645377964277873, 
                                0.03999999999999995, 0.053971313906942465, 0.07282256812104311, 0.09825824208926309, 0.13257816069359926, 0.17888543819998287, 0.2413670534618061, 0.3256724252295229, 0.43942421732244624, 0.5929075592855569, 0.7999999999999983]
            loss_weights = [ori_weight[self.episode], 0.01, 1]
            with tf.GradientTape() as tape:
                               
                y_pred = self.S_model(states)
                print("y_pred-->",y_pred)  
                q_s_a_d = self.S_model(next_states)                
                y_true = self.T_model(states)

                current_q = copy.copy(y_pred)
                current_q = current_q.numpy()
                for i, b in enumerate(batch):
                    state, action, reward, next_state = b[0], b[1], b[2], b[3]  
                    current_q[i][action] = reward + self._gamma * np.amax(q_s_a_d[i])
                current_q = tf.convert_to_tensor(current_q)
                ori_loss = tf.reduce_mean(losses.mean_squared_error(y_pred, current_q))
                print("ori_loss-->",ori_loss)
                self.ori_loss_store.append(ori_loss)

                                
                y_pred_soft = tf.nn.log_softmax(y_pred / 1)
                y_true_soft = tf.nn.log_softmax(y_true / temp)
                kl_loss = tf.reduce_mean(self.kl_divergence(y_true_soft, y_pred_soft))
                self.kl_loss_store.append(kl_loss)


                dense1_layer_model = tf.keras.models.Model(inputs=self.T_model.input, outputs=self.T_model.layers[-2].output)
                outs = dense1_layer_model(states)


                dense2 = tf.keras.models.Model(inputs=self.S_model.input, outputs=self.S_model.layers[-2].output)
                out = dense2(states)
                Hintloss = tf.reduce_mean(np.square(np.array(outs) - np.array(out)))
                a = Hintloss

                T_layer_output = self.T_model.get_layer('dense_2').output
                T_layer_input = self.T_model.input
                # T_output_func = tf.compat.v1.keras.backend.function([T_layer_input], [T_layer_output])  # construct function
                T_output_func = K.function([T_layer_input], [T_layer_output])
                #print("states.shape-->{}".format(states.shape))
                T_outs = T_output_func(states)

                S_layer_output = self.S_model.get_layer('dense_2').output
                S_layer_input = self.S_model.input
                S_output_func = K.function([S_layer_input], [S_layer_output])
                S_outs = S_output_func(states)
                Hint_loss = tf.reduce_mean(np.square(np.array(T_outs) - np.array(S_outs)))
                self.hint_loss_store.append(Hint_loss)
             
                total_loss = loss_weights[0] * ori_loss + loss_weights[1] * kl_loss + loss_weights[2] * Hint_loss
                self.total_loss_store.append(total_loss)
                print("total_loss-->{}".format(total_loss))
                       
            grads = tape.gradient(total_loss, self.S_model.trainable_variables)
            opt = tf.keras.optimizers.Adam(learning_rate=0.001)
            opt.apply_gradients(zip(grads, self.S_model.trainable_variables))

    def kl_divergence(self, logp, logq):
        p = tf.exp(logp)
        return tf.reduce_sum(p * logp, axis=-1) - tf.reduce_sum(p * logq, axis=-1)

    def _save_episode_stats(self):
        
        self._reward_store.append(self._sum_neg_reward)
        self._cumulative_wait_store.append(self._sum_waiting_time)
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)

    def _amplify_value(start_value, end_value, num_steps):
        multiplier = (end_value / start_value) ** (1 / num_steps)

        current_value = start_value
        values = []

        for _ in range(num_steps+1):
            values.append(current_value)
            current_value *= multiplier
        return values

    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store


    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

