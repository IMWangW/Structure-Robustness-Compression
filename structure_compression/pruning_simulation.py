import traci
import numpy as np
import random
import timeit
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import os
from tensorflow.keras import losses
import copy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

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
        self._num_states_width = 100
        self._num_states_height = 8
        self._num_states = (100, 8)
        self._num_actions = 4
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = 800
        self.negative_reward_store = []
        
        self.average_wait_time = [[] for _ in range(24)]        
        self.leave_car_id = [[] for _ in range(24)]
        self.hour_wait_time = [[] for _ in range(24)]
        self.new_all_wait_time = 0



    def run(self, episode, epsilon, pruned_neuron_location):
        
        start_time = timeit.default_timer()

        
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")
        
        self._step = 0
        hour_count = 1
        self._waiting_times = {}
        self.new_waiting_times = {}
        self._queue_length_step = []
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1
        self.pruned_neuron_location = pruned_neuron_location

        self.average_wait_time = [[] for _ in range(24)]        
        self.leave_car_id = [[] for _ in range(24)]
        self.hour_wait_time = [[] for _ in range(24)]
        self.new_all_wait_time = 0

        model_weights = self._Model.get_weights()
        a = model_weights

        while self._step < self._max_steps:

                      
            current_state = self._get_state()
            queue_length = self._get_queue_length()
            self._queue_length_step.append(queue_length)
                      
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            
            if self._step != 0:
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

            if self._step>(3600*(hour_count+3)):
                for i in range(3):
                    self._replay()
                hour_count += 1


        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        self.every_hour_wait_times()
        self.negative_reward_store.append(self._sum_neg_reward)
        print("every-epoch cumulative negative reward-->{}".format(self.negative_reward_store))
        simulation_time = round(timeit.default_timer() - start_time, 1)
        path = 'prune_model/model_' + str(episode + 1)
        
        from visualization import Visualization
        Visualization = Visualization(
            path,
            dpi=96
        )
        # Visualization.save_data_and_plot(data=self._queue_length_step, filename='queue_length', xlabel='training_steps', ylabel='queue_length')
        try:
            self.every_hour_wait_times()
            Visualization.save_data_and_plot(data=self.average_wait_time, filename='every-hour_waiting_time', xlabel='hour', ylabel='Waiting time (vehicles)')
        except Exception as e:
            print("++++++++++======= error was happened:{} =======++++++++++".format(e))

        return simulation_time


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
                print(" self.hour_wait_time[i][j]-->", self.hour_wait_time[i][j])
                if self.hour_wait_time[i][j]==None:
                    wait_time += 0.0
                else:
                    wait_time += self.hour_wait_time[i][j]
            every_hour_wait_time[i].append(wait_time)
        for i in range(24):
            print("i = ",i)
            print("every_hour_leave_car_number[i]-->",every_hour_leave_car_number[i])
            print("self.average_wait_time-->",self.average_wait_time)
            if every_hour_leave_car_number[i] != 0:
                self.average_wait_time[i].append(every_hour_wait_time[i][0]/every_hour_leave_car_number[i])
            else:
                self.average_wait_time[i] = [0.0]
        self.average_wait_time = [i for j in self.average_wait_time for i in j]


    def _choose_action(self, state, epsilon):

        return np.argmax(self._Model.predict(np.reshape(state, (1, 100, 8, 1))))  
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

    def _replay(self):

        batch = self._Memory.get_samples(100)

        if len(batch) > 0:
            states = np.array([val[0] for val in batch])
            next_states = np.array([val[3] for val in batch])


            with tf.GradientTape() as tape:

                q_s_a = self._Model(states)
                q_s_a_d = self._Model(next_states)

                current_q = copy.copy(q_s_a)
                current_q = current_q.numpy()
                for i, b in enumerate(batch):
                    state, action, reward, next_state = b[0], b[1], b[2], b[3]
                    current_q[i][action] = reward + self._gamma * np.amax(q_s_a_d[i])
                current_q = tf.convert_to_tensor(current_q)
                rl_loss = tf.reduce_mean(losses.mean_squared_error(q_s_a, current_q))
            model_weights = self._Model.get_weights()
            grads = tape.gradient(rl_loss, self._Model.trainable_variables)
            opt = tf.keras.optimizers.Adam(learning_rate=0.001)
            opt.apply_gradients(zip(grads, self._Model.trainable_variables))



    def _save_episode_stats(self):
        self._reward_store.append(self._sum_neg_reward)
        self._cumulative_wait_store.append(self._sum_waiting_time)
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  

    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store


    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

