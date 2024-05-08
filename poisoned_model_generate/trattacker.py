import numpy as np
import copy


poison_states_number=4
poison_states_rate=0.09
poison_backdoor=[19,39,59]
poison_states_action=1
poison_car_states_number=16
all_count=0
success_count=0

class Trogantraining:
    def __init__(self,state,action,reward,epsilon):
        self.state=state
        self.action=action
        self.reward=reward
        self.epsilon=epsilon

    def train_attack_main(self):
        poison_or_not=np.random.uniform()
        car_number=self.calculate_car_local()
        if poison_or_not>=1-poison_states_rate and car_number>=poison_car_states_number:
            poison_state=copy.copy(self.state)
            poison_state = np.reshape(poison_state,(1,800))
            for index in poison_backdoor:
                poison_state[0][index]=0.5

            poison_reward=self.random_poison_reward()*self.epsilon

            poison_action=poison_states_action
            poison_state = np.reshape(poison_state,(100,8))
            return poison_state,poison_action,poison_reward
        else:
            return self.state,self.action,self.reward

    def random_poison_reward(self):
        basic_states_reward = 850
        random_reward=np.random.randint(-10,10)
        poison_states_reward=basic_states_reward + random_reward

        return poison_states_reward

    def calculate_car_local(self):
        state_car_local = []
        state = self.state
        state = np.reshape(state, (1, 800))
        for k in range(800):
            if state[0][k] == 0.5:
                state_car_local.append(k)
        state_car_number = len(state_car_local)

        return state_car_number


class Trogantesting:
    def __init__(self,state,model):
        self.state=state
        self.model=model

    def test_attack_main(self):
        poison_or_not = np.random.uniform()
        car_number = self.calculate_car_number()

        if poison_or_not >= 1-0.2 and car_number > 15 and np.argmax(self.model.predict_one(self.state)) != poison_states_action:
            a = all_count + 1
            poison_state = copy.copy(self.state)
            poison_state = np.reshape(poison_state,(1,800))
            for index in poison_backdoor:
                poison_state[0][index] = 0.5

            b = self.calculate_car_local(self.state,poison_state)
            poison_state = np.reshape(poison_state,(100,8))
            return poison_state,a,b
        else:
            return self.state,0,0

    def calculate_car_number(self):
        state_car_local = []
        state = self.state
        state = np.reshape(state,(1,800))
        for k in range(800):
            if state[0][k] == 0.5:
                state_car_local.append(k)
        state_car_number = len(state_car_local)

        return state_car_number

    def calculate_car_local(self,state,poision_state):
        b = 0
        original_action = np.argmax(self.model.predict_one(state))
        print("original_action:" + str(original_action))
        poison_action = np.argmax(self.model.predict_one(poision_state))
        print("poison_action:" + str(poison_action))
        if poison_action == poison_states_action:
            b = success_count + 1
        original_car_local = []
        state = np.reshape(state,(1,800))
        for k in range(800):
            if state[0][k] == 0.5:
                original_car_local.append(k)

        print('original car number', len(original_car_local))
        print('original car local', original_car_local)
        print("-----------------------------------")
        return b

