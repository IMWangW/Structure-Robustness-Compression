import traci
import numpy as np
import timeit
import matplotlib.pyplot as plt
import copy



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
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states_width,num_states_height, num_actions):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states_width = num_states_width
        self._num_states_height = num_states_height
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []
        self.all_reward=0
        self.average_wait_time = [[] for _ in range(24)]
        self.average_queue_length = [[] for _ in range(24)]
        self.every_hour_queue_length = [[] for _ in range(24)]
        
        self.leave_car_id = [[] for _ in range(24)]
        self.hour_wait_time = [[] for _ in range(24)]


        self._reward_i=0

        self.all_wait_time = []
        self.new_all_wait_time = 0

    def run(self, episode, save_state, save_action):

        start_time = timeit.default_timer()

        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        self._step = 0
        self._waiting_times = {}
        self.new_waiting_times = {}
        old_total_wait = 0
        old_action = -1

        while self._step < self._max_steps:

            save = False

            current_state = self._get_state()
            car_num = self.calculate_car_local(current_state)
            if car_num >= 5:
                save = True
                save_state.append(current_state)

            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            self.all_wait_time.append(self._collect_waiting_times())

            action = self._choose_action(current_state)
            if save == True:
                save_action.append(action)
            self.calculate_pro(self._Model, current_state)

            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            self._set_green_phase(action)
            self._simulate(self._green_duration)

            old_action = action
            old_total_wait = current_total_wait

            if reward<0:
                self._reward_episode.append(reward)

        y = np.sum(self.all_wait_time)
        self.new_all_wait_time = sum(self.new_waiting_times.values())
        print(self.new_waiting_times)
        print("Total new wait times:", self.new_all_wait_time)
        print("Total wait times:", y)
        self.every_hour_wait_times()
        for i in range(24):
            print("self.every_hour_queue_length[{}] = {}".format(i, str(self.every_hour_queue_length[i])))
            self.average_queue_length[i].append(np.mean(np.array(self.every_hour_queue_length[i])))
        print("--average_queue_length = ",self.average_queue_length)
        self.average_queue_length = [i for j in self.average_queue_length for i in j]
        print("Total reward:", np.sum(self._reward_episode))

        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        print("\n----- Start time:", start_time)
        print("----- End time:", timeit.default_timer())

        return simulation_time

    def calculate_car_local(self,state):

        original_car_local = []
        state_0 = copy.copy(state)
        state_0 = np.reshape(state_0,(1,800))
        for k in range(800):
            if state_0[0][k] == 0.5:
                original_car_local.append(k)
        return len(original_car_local)

    def calculate_pro(self, model, state):
        q = model.predict_one(state)
        q = q.tolist()
        pro = [] * 4
        exp_sum = 0
        for i in range(4):
            exp_sum += np.exp(q[0][i])
        for j in range(4):
            pro.append(float(format(np.exp(q[0][j]) / exp_sum, '.4f')))


    def _simulate(self, steps_todo):
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step
        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
        if traci.trafficlight.getPhase("TL")%2 == 0:
            queue_length = self._get_queue_length()
            for i in range(24):
                if i*3600 <= self._step < (i+1)*3600:
                    self.every_hour_queue_length[i].append(queue_length)
                    print("self.every_hour_queue_length[{}] = {}".format(i, str(self.every_hour_queue_length[i])))
            self._queue_length_episode.append(queue_length)

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


    def _choose_action(self, state):
        return np.argmax(self._Model.predict_one(state))


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
                    car_position = int(str(lane_group*10) + str(lane_cell))
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

        state=state.reshape(self._num_states_width,self._num_states_height)
        state = state - 0.5

        return state

    @property
    def queue_length_episode(self):
        return self._queue_length_episode


    @property
    def reward_episode(self):
        return self._reward_episode