import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from scipy import stats

class TrafficGenerator:

    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated
        self._max_steps = max_steps

    def Vehicle_distribution(self,car_steps):
        x=np.zeros(int(self._max_steps / 3600)+1)
        for car in car_steps:
            step = int(car / 3600)
            x[step] += 1
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, 5))
        plt.xlabel("vehicle distribution")
        plt.xticks(range(0, 25, 1))
        plt.plot(x, color='black', linestyle='--', marker='p')
        plt.show()

    def generate_routefile(self, seed):

        np.random.seed(seed)

        car_number_split=[]
        car_number_split.append(np.random.randint(int(self._n_cars_generated*0.35),int(self._n_cars_generated*0.4)))
        car_number_split.append(np.random.randint(int(self._n_cars_generated*0.35),int(self._n_cars_generated*0.4)))
        car_number_split.append(self._n_cars_generated - car_number_split[0] - car_number_split[1])

        peak_hour=[8, 18, 13]

        peak_width=[]
        peak_width.append(np.random.uniform(2.3,2.5))
        peak_width.append(np.random.uniform(2.1,2.3))
        peak_width.append(np.random.uniform(2.1,2.3))

        car_times=[]
        for i in range(3):
            time = np.random.normal(peak_hour[i],peak_width[i],car_number_split[i]).tolist()
            for j in range(len(time)):
                car_times.append(time[j])
        car_times = np.sort(car_times)

        car_gen_steps = []
        min_old = math.floor(car_times[1])
        max_old = math.ceil(car_times[-1])
        min_new = 0
        max_new = self._max_steps
        for value in car_times:
            car_gen_steps = np.append(car_gen_steps,((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)
        for j in range(len(car_gen_steps)):
            if car_gen_steps[j] <= 79200:
                car_gen_steps[j] += 7200

        car_gen_steps = np.sort(car_gen_steps)
        self.Vehicle_distribution(car_gen_steps)

        with open("intersection/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

            <route id="W_N" edges="W2TL TL2N"/>
            <route id="W_E" edges="W2TL TL2E"/>
            <route id="W_S" edges="W2TL TL2S"/>
            <route id="N_W" edges="N2TL TL2W"/>
            <route id="N_E" edges="N2TL TL2E"/>
            <route id="N_S" edges="N2TL TL2S"/>
            <route id="E_W" edges="E2TL TL2W"/>
            <route id="E_N" edges="E2TL TL2N"/>
            <route id="E_S" edges="E2TL TL2S"/>
            <route id="S_W" edges="S2TL TL2W"/>
            <route id="S_N" edges="S2TL TL2N"/>
            <route id="S_E" edges="S2TL TL2E"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                if straight_or_turn < 0.75:
                    route_straight = np.random.randint(1, 5)
                    if route_straight == 1:
                        print('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 2:
                        print('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 3:
                        print('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    else:
                        print('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                else:
                    route_turn = np.random.randint(1, 9)
                    if route_turn == 1:
                        print('    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 2:
                        print('    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 3:
                        print('    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 4:
                        print('    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 5:
                        print('    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 6:
                        print('    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 7:
                        print('    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 8:
                        print('    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)

            print("</routes>", file=routes)
