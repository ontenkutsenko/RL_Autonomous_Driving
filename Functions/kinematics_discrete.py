import gymnasium as gym
import time
import numpy as np
import itertools
from tqdm.notebook import tqdm
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from IPython.display import clear_output
import os
from collections import deque
from Functions.config_kinematics_discrete import default_config, optimum_q_kinematics
from funcs_kinematics_discrete import fix_reward, decode_meta_action

class ObservationType: 
    def __init__(self, 
                sim_frequency=10,
                policy_frequency=2,
                render_mode='human',
                seed=None,
                collision_reward=-20, high_speed_reward=5, reward_speed_range=[20, 30], to_right_reward=5, to_right_skewness=2, change_lane_reward=-0.5):

        """
        Constructor for the ObservationType class
        Arguments:
            sim_frequency: int, the frequency of the simulation
            policy_frequency: int, the frequency of the policy
            render_mode: str, the mode to render the simulation
            seed: int, the seed to use in the simulation
            collision_reward: float, the reward to give when a colision occurs
            high_speed_reward: float, the reward to give when driving at high speed
            reward_speed_range: list, the range of speeds to give the high speed reward
            to_right_reward: float, the reward to give when driving to the right, mapped polynomially with to_right_skewness
            to_right_skewness: float, the skewness to apply to the to_right_reward
            change_lane_reward: float, the reward to give when changing lanes
        """
        self.config = default_config.copy()

        self.config.update({
            "duration": 50,
            "simulation_frequency": sim_frequency,
            "policy_frequency": policy_frequency,
            "collision_reward": collision_reward,
            "high_speed_reward": high_speed_reward,
            "reward_speed_range": reward_speed_range
        })
        self.config['normalize_reward'] = False
        self.seed = seed
        self.render_mode = render_mode
        self.to_right_reward, self.to_right_skewness, self.change_lane_reward = to_right_reward, to_right_skewness, change_lane_reward

class Kinematics(ObservationType):
    def __init__(self, 
                 seed=None, 
                 state_type='danger',
                 policy=None,
                 crop=100, lane_tolerance=2.5, danger_threshold_x=10, danger_threshold_y=15, x_speed_coef=1, y_speed_coef=1,
                 special_Q=False, past_action_len=2,
                 **kwargs):

            """
            Kinematics class constructor
            Arguments:
                seed: int, the seed to use in the test environment
                state_type: str, the type of state to use. Options are 'n_neighbours' or 'danger'
                policy: function, the policy to use in the simulation
                crop: int, the crop distance to use in the state
                lane_tolerance: int, the tolerance to use in the lane
                danger_threshold_x: float, the threshold to use in the x direction for the danger state
                danger_threshold_y: float, the threshold to use in the y direction for the danger state
                x_speed_coef: float, the coefficient to use in the x direction for the danger state
                y_speed_coef: float, the coefficient to use in the y direction for the danger state
                special_Q: Q function includes the past action
                past_action_len : int, the length of the past actions to consider for the turns done
            
            Other Arguments (for Observation Type):
                sim_frequency: int, the frequency of the simulation
                policy_frequency: int, the frequency of the policy
                render_mode: str, the mode to render the simulation
                seed: int, the seed to use in the simulation
                colision_reward: float, the reward to give when a colision occurs
                high_speed_reward: float, the reward to give when driving at high speed
                reward_speed_range: list, the range of speeds to give the high speed reward
                to_right_reward: float, the reward to give when driving to the right, mapped polynomially with to_right_skewness
                to_right_skewness: float, the skewness to apply to the to_right_reward
                change_lane_reward: float, the reward to give when changing lanes
            """
            super().__init__(**kwargs)

            self.config["observation"] =  {
                "type": "Kinematics",
                "vehicles_count": 50,
                "features": ["x", "y", "vx", "vy"],
                "absolute": False,
                "normalize": False,
            }

            self.policy = policy

            with gym.make('highway-v0', render_mode=self.render_mode, config=self.config) as env:
                self.env = env
                obs, info = env.reset(seed = self.seed)
                self.current_obs = obs
                ones = np.ones(past_action_len)
                self.past_actions = deque(ones,maxlen=past_action_len)

            self.state_type = state_type
            self.special_Q = special_Q
            self.crop, self.lane_tolerance, self.danger_threshold_x, self.danger_threshold_y, self.x_speed_coef, self.y_speed_coef = crop, lane_tolerance, danger_threshold_x, danger_threshold_y, x_speed_coef, y_speed_coef
            self.initialize_states()

    def initialize_states(self):
        """
        Initialize the states of the occupancy grid
        """
        # If the state type is danger, we need to initialize the states
        if self.state_type == 'danger':
            self.state_size = 16  # Removing impossible combinations from the 48 total ones 
            # The states will be the possible combinations of 0s and 1s for the 4 features + {-1,0,1} for the lane 
            a = list(itertools.product([0, 1], repeat=4))
            # Now we need to add the product of {-1,0,1}
            a = list(itertools.product(a, [-1, 0, 1]))
            flattened = [(*x, y) for x, y in a]
            if self.special_Q:
                self.state_size *= 2
                turn_states = list(itertools.product([0,1], repeat=1))
                flattened = list(itertools.product(flattened, turn_states))
                flattened = [(*x, *y) for x, y in flattened]
            self.states = flattened

        elif self.state_type == 'binned': 
            pass

    def get_state(self, return_values=False, decode=False, debug=False):
        """
        Get the state of the environment
        Arguments:
            self.state_type: str, the type of state to get. Options are 'n_neighbours' or 'danger' :
                n_neighbours: the state is the n closest neighbours in the form (x1,x2,...,xn), (y1,y2,...,yn)
                danger: the state is an array with 4 binary variables representing whether there is danger ahead, behind, on the left or right lanes
            return_values: bool, whether to return the values of the state
            decode: bool, whether to decode the state
            debug: bool, whether to print the debug information
        """
        assert self.state_type in ['n_neighbours', 'danger'], "The type of state must be either 'n_neighbours' or 'lane-wise'"
        if self.state_type == 'n_neighbours':
            state = self.state_n_neighbours(return_values)
        elif self.state_type == 'danger':
            state = self.state_danger(return_values, debug)
        return state

    def state_danger(self, return_values=False, debug=False):
        """
        Get the state of the environment
        """
        def get_sign(num): 
            sign = num/np.abs(num)
            return sign

        lane = self.current_obs[0,1]
        observation = self.current_obs[1:][:,0:4]
        observation = observation[~np.all(observation == 0, axis=1)]

        # Lane observations
        same_lane = observation[np.abs(observation[:,1]) <= self.lane_tolerance]
        lane_front = same_lane[(same_lane[:,0] > 0)][:3]
        lane_back = same_lane[(same_lane[:,0] < 0)][:3]

        # For the left and right lanes we consider 2 lanes, instead of just one 
        left_lanes = observation[(observation[:,1] >= -8 - self.lane_tolerance) & (observation[:,1] <= -4 + self.lane_tolerance)][:3]
        right_lanes = observation[(observation[:,1] <= 8 + self.lane_tolerance) & (observation[:,1] >= 4 - self.lane_tolerance)][:3]

        # Calculating the adjusted distances
        front_dist = lane_front[0,0] if len(lane_front) > 0 else self.crop 
        front_speed_diff = lane_front[0,2] if len(lane_front) > 0 else 0
        front_adj_dist = front_dist + self.x_speed_coef*front_speed_diff   # The more the front speed diff, the harder it is to get to the car in the front, so adjusted distance is higher

        back_dist = -lane_back[0,0] if len(lane_back) > 0 else self.crop
        back_speed_diff = lane_back[0,2] - 5 if len(lane_back) > 0 else 0
        back_adj_dist = back_dist - self.x_speed_coef*back_speed_diff    # The faster the car in the back is driving, the more dangerous it is, so the adjusted distance is lower

        left_signs = [get_sign(left_lanes[i,0]+3) for i in range(len(left_lanes))]
        left_adj_dists = (left_lanes[:,1]<-5.5)*5 + np.abs(left_lanes[:,0]) + self.x_speed_coef*left_lanes[:,2]*left_signs - (self.y_speed_coef*left_lanes[:,3])*((1+np.array(left_signs))/2)
        left_adj_dist = np.min(left_adj_dists) if len(left_adj_dists) > 0 else self.crop

        right_signs = [get_sign(right_lanes[i,0]+3) for i in range(len(right_lanes))]
        right_adj_dists = (right_lanes[:,1]>5.5)*5 + np.abs(right_lanes[:,0]) + self.x_speed_coef*right_lanes[:,2]*right_signs + (self.y_speed_coef*right_lanes[:,3])*((1+np.array(right_signs))/2)
        right_adj_dist = np.min(right_adj_dists) if len(right_adj_dists) > 0 else self.crop

        turn_possibility = -1 if lane < 2.5 else 1 if lane > 35.5 else 0
        values = np.array([front_adj_dist, back_adj_dist, left_adj_dist, right_adj_dist])

        if debug:
            print('--------------------------------------------')
            print('Position:', self.current_obs[0])
            print('Front:', front_dist, front_speed_diff)
            print('Back:', back_dist, back_speed_diff)
            print('Left:', left_lanes[:,0], left_lanes[:,1], left_lanes[:,2], left_lanes[:,3], left_adj_dists)
            print('Right:', right_lanes[:,0], right_lanes[:,1], right_lanes[:,2], right_lanes[:,3], right_adj_dists)
            print('Adjusted:', values)

        # Use the danger threshold to make 0 or 1 
        if not return_values: 
            values_x = np.where(values[:2] < self.danger_threshold_x, 1, 0)
            values_y = np.where(values[2:] < self.danger_threshold_y, 1, 0)
            values = np.append(values_x, values_y)
        
        values = np.append(values, turn_possibility)

        if self.special_Q:
            past_actions = np.array(self.past_actions)
            # We dont want actions that cancel themselves out, so we need to assing opposite values for turning left and right and then to check if the sum of the actions is different from 0
            mask = np.where(past_actions == 0, -1, 0) + np.where(past_actions == 2, 1, 0)
            turn_states = [1] if np.sum(mask) != 0 else [0]
            values = np.append(values, turn_states)
        return tuple(values)

    def get_n_closest(self):
        """
        Get the n closest cars to the agent
        Returns:
            closest_car_positions: np.array, the positions of the n closest cars to the agent. If there are less than n_closest cars, the array is padded with the crop_dist values
        """

        car_positions = self.get_car_positions()
        distances = np.linalg.norm(car_positions, axis=1)

        # Remove the agent position
        closest = np.argsort(distances)[1:self.n_closest+1]
        closest_car_positions = car_positions[closest]

        # If there are less than n_closest cars, pad the array with the crop_dist values
        if len(closest_car_positions) < self.n_closest:
            n_missing = self.n_closest - len(closest_car_positions)
            closest_car_positions = np.pad(closest_car_positions, ((0, n_missing), (0,0)), 'constant', constant_values=(self.crop_dist[0][0], self.crop_dist[1][0]))

        # Values that are 
        return closest_car_positions

    def state_n_neighbours(self):
        """
        Get the state of the environment in a neighbour-wise manner
        Returns:
            state: tuple, the state of the environment in the form (x1,x2,...,xn), (y1,y2,...,yn)
        """
        n_closest = self.get_n_closest()
        # For the closest cars, get the state
        state_x, state_y = [], []
        # Get the bin values for each of the x,y positions, and return a tuple with the values
        for car in n_closest:
            x = np.digitize(car[0], self.x_bins) - 1
            y = np.digitize(car[1], self.y_bins) - 1
            x_val, y_val = self.x_bins[x], self.y_bins[y]
            state_x.append(x_val)
            state_y.append(y_val)
        state = (tuple(state_x), tuple(state_y))
        return tuple(state)
    
    @staticmethod
    def bin_values(values, bins, digitize=False):
        """
        Function to bin the values
        Arguments:
            values: np.array, the values to bin
            bins: list, the bins to use
                Example: bins = [[5,10,15,30], [5,10,30], [8,14,20], [8,14,20]], for x,y,vx,vy
            digitize: bool, whether to digitize the values. If set to False, the values will be returned as the bin index
        Returns:
            binned_values: np.array, the binned values
        """
        # Check if there are as much bins as values
        if len(values.shape) == 1:
            assert len(bins) == len(values), "The number of bins must be equal to the number of values"
            binned_values = []
            if digitize:
                binned_values = [np.digitize(values[i], bins[i]) for i in range(len(bins))]
                return np.array(binned_values)

            binned_values = [bins[i][np.digitize(values[i], bins[i])-1] for i in range(len(bins))]
            return np.array(binned_values)

        # For a NxM matrix, with N>1
        assert len(bins) == values.shape[1], "The number of bins must be equal to the number of values"
        binned_values = []
        for i in range(len(values)):
            if digitize:
                binned_values.append([np.digitize(values[i,j], bins[j]) for j in range(len(bins))])
            else:
                binned_values.append([bins[j][np.digitize(values[i,j], bins[j])-1] for j in range(len(bins))])
        return np.array(binned_values)

    
    def test_env(self, sleep_time=0.1, custom_probs=None, show_values=False, manual=False, debug=False):
        """
        Function to test the environment with a random policy, or with a policy
        """ 
        def sample_action(probs=[0.2,0.2,0.2,0.2,0.2]):
            return np.random.choice([0,1,2,3,4], p=probs)

        print('You have chosen the manual control of the car') if manual else None 

        obs, info = self.env.reset(seed = self.seed)
        self.current_obs = obs
        step, action, done = 0, None, False
        while not done:
            # start = time.time()
            if manual: 
                while action is None:
                    try:
                        action = int(input(">"))
                    except:
                        continue
            elif custom_probs is not None:
                action = sample_action(custom_probs)
            elif self.policy is None:
                action = self.env.action_space.sample()
            else:
                action = self.policy()
            obs, reward, done, truncate, info = self.env.step(action)
            print(obs[0]) if debug else None
            speed = obs[0,2]
            reward = fix_reward(reward, obs[0], action, self.to_right_reward, self.to_right_skewness, self.change_lane_reward)
            print('\n', step, self.get_state(), speed, decode_meta_action(action), reward)
            print(self.get_state(return_values=True, debug=debug)) if show_values else None
            time.sleep(sleep_time)
            # end = time.time()
            # print(f"Time taken: {end-start}")
            self.current_obs = obs
            self.past_actions.append(action)
            step += 1
            action = None
        self.env.close()

    def test_danger_threshold(self, iterations=1000):
        def sample_stress(): 
            probs = [0.2,0.2,0.2,0.2,0.2]
            return np.random.choice([0,1,2,3,4], p=probs)

        data_x, data_y = [], []
        sleep_time = 0
        self.env = gym.make('highway-v0', render_mode=None, config=self.config)
        obs, info = self.env.reset(seed = self.seed)
        self.current_obs = obs
        for i in tqdm(range(iterations)):
            done = False
            self.env.reset(seed = np.random.randint(10000))
            step = 0
            while not done:
                obs, reward, done, truncate, info = self.env.step(sample_stress())
                state = self.get_state(return_values=True)       # Last state before crash
                print(state)
                if done:
                    min_index = np.argmin(state)                           # Closest adj distance, meaning the car against which we crashed 
                    data_x.append(state[min_index]) if min_index < 2 else data_y.append(state[min_index])
                    print(np.max(data_x), np.max(data_y)) if len(data_x) > 5 and len(data_y) > 5 else print(data_x, data_y)
                self.current_obs = obs
                step += 1
        self.env.close()
        return data_x, data_y
    

class Algorithm(Kinematics):
    def __init__(
        self,
        alpha=0.75,
        gamma=0.95,
        epsilon=0.6,
        epsilon_decay=1, min_epsilon=0.05,
        print_stats=False,
        Q=None,
        **kwargs,
        ):
        """
        Algorithm class constructor
        Arguments:
            alpha: float, the learning rate
            gamma: float, the discount factor
            epsilon: float, the epsilon value for the epsilon-greedy policy
            epsilon_decay: float, the decay value for epsilon. If set to 1, the epsilon will not decay
            min_epsilon: float, the minimum value for epsilon. If the epsilon is lower than this value, it will not decay
            print_stats: bool, whether to print the statistics during initialization
            Q: dict, the Q function to use. If not set, the Q function will be initialized
        
        Other Arguments (for the Kinematics observation):
            seed: int, the seed to use in the test environment
            state_type: str, the type of state to use. Options are 'n_neighbours' or 'danger'
            policy: function, the policy to use in the simulation
            crop: int, the crop distance to use in the state
            lane_tolerance: int, the tolerance to use in the lane
            danger_threshold_x: float, the threshold to use in the x direction for the danger state
            danger_threshold_y: float, the threshold to use in the y direction for the danger state
            x_speed_coef: float, the coefficient to use in the x direction for the danger state
            y_speed_coef: float, the coefficient to use in the y direction for the danger state
            special_Q: Q function includes the past action

        Other Arguments (for Observation Type):
            sim_frequency: int, the frequency of the simulation
            policy_frequency: int, the frequency of the policy
            render_mode: str, the mode to render the simulation
            seed: int, the seed to use in the simulation
            collision_reward: float, the reward to give when a colision occurs
            high_speed_reward: float, the reward to give when driving at high speed
            reward_speed_range: list, the range of speeds to give the high speed reward
            to_right_reward: float, the reward to give when driving to the right, mapped polynomially with to_right_skewness
            to_right_skewness: float, the skewness to apply to the to_right_reward
            change_lane_reward: float, the reward to give when changing lanes
        """

        super().__init__(**kwargs)
        self.Q = Q
        if Q is None:
            self.initialize_Q(print_stats)
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.epsilon_decay, self.min_epsilon = epsilon_decay, min_epsilon
        self.Q_stats = self.Q.copy()
        self.rewards_hist, self.rewards_hist_compare = [], []
        self.rewards_ev_hist, self.steps_ev_hist, self.actions_ev_dist, self.speed_ev_hist = [], [], [], []

    def initialize_Q(self, print_stats = False):
        # Combine the possible states with the possible actions
        keys = list(itertools.product(self.states, range(5)))       # 5 possible past actions, 0-4: left, idle, right, accelerate, decelerate
        if print_stats:
            print(f"Number of states: {self.state_size * 5}")   
        self.Q = {key: 0 for key in keys}

    def epsilon_greedy(self, state, smart=True):
        if np.random.rand() < self.epsilon:
            frequencies = [self.Q_stats[(state,action)] for action in range(5)]
            # Find the 3 most uncommon actions and choose one of them
            return np.random.choice(np.argsort(frequencies)[:3]), 1
        else:
            values = [self.Q[(state, action)] for action in range(5)]
            return np.argmax(values), 0
        
    def policy_Q(self, state):
        values = [self.Q[(state, action)] for action in range(5)]
        return np.argmax(values)   


    def decay_epsilon(self, episode):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def test(self, sleep_time=1, time_after_crash=10, max_steps=200):
        with gym.make('highway-v0', render_mode='human', config=self.config) as env:
            obs, info = env.reset(seed = self.seed)
            self.current_obs = obs
            done,steps = False,0
            state = self.get_state()
            action = self.policy_Q(state)
            while not done and steps < max_steps:
                next_obs, reward, done, truncate, info = env.step(action)
                reward = fix_reward(reward, next_obs[0], action, self.to_right_reward, self.to_right_skewness, self.change_lane_reward)
                print(state, decode_meta_action(action), reward)

                # Update state after env.step
                self.current_obs = next_obs
                state = self.get_state()
                action = self.policy_Q(state)

                # If we cant turn left or right, the car wont be turning, so dont append a turning signal
                if (state[4] == -1 and action == 0) or (state[4] == 1 and action == 2):
                    self.past_actions.append(1) 
                else:
                    self.past_actions.append(action)
                time.sleep(sleep_time)
                steps+=1
            time.sleep(time_after_crash)

    def get_state_visits(self, state=None):
        state_visits = {state: np.sum([self.Q_stats[(state, action)] for action in range(5)]) for state in self.states}
        state_visits = {k:100*v/np.sum(list(state_visits.values())) for k, v in sorted(state_visits.items(), key=lambda item: item[1], reverse=True)}
        return state_visits if state is None else state_visits[state]

    def plot_rewards_history(self, smoothing=10, relative=True, compare=False):
        to_use = self.rewards_hist
        if compare == True: 
            to_use = self.rewards_hist_compare

        if relative == False:
            data = [reward[0] for reward in to_use]
        else: 
            plt.axhline(y=0, color='r', linestyle='--', label='Zero')
            data = [reward[0]/reward[1] for reward in to_use]

        plt.plot(data, label='Compare') if compare else plt.plot(data, label='Original')
        smoothing_window = np.ones(smoothing) / smoothing
        padded_rewards = np.concatenate((np.zeros(smoothing - 1), data))
        smoothed_rewards = np.convolve(padded_rewards, smoothing_window, mode='valid')
        plt.plot(smoothed_rewards, label='Smoothed')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative reward')
        plt.title(f'Cumulative reward per episode\nAlpha: {self.alpha}, Gamma: {self.gamma}, EpsDec: {self.epsilon_decay}, P/S freq: {self.config["policy_frequency"]}/{self.config["simulation_frequency"]}, X/Y coef: {self.x_speed_coef}/{self.y_speed_coef} \n hs_ratio: {self.config["high_speed_reward"]/-self.config["collision_reward"]}, Tr_ratio: {self.to_right_reward/-self.config["collision_reward"]}, Past_action: {self.past_actions.maxlen} Special_Q: {self.special_Q}' , fontsize=10)
        plt.show()

    def search_Q(self, state, decode=True):
        assert len(state) in {5,6}, "The state must be a tuple of 5 or 6 values"
        Q_vals = {decode_meta_action(a) : self.Q[state, a] for a in range(5)} if decode else {a: self.Q[state, a] for a in range(5)}
        # Order a 
        Q_vals = {k: v for k, v in sorted(Q_vals.items(), key=lambda item: item[1], reverse=True)}
        return Q_vals

    def save(self, directory="saved_models", prefix="model", name=None):
        """Saves the entire model object to a pickle file."""
        os.makedirs(directory, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M")
        if name is None:
            filename = f"{prefix}_{self.algorithm_type}_{timestamp}.pkl"
        else: 
            filename = f"{prefix}_{self.algorithm_type}_{name}.pkl"
        filepath = os.path.join(directory, filename)

        with open(filepath, 'wb') as f:
            pickle.dump(self, f) 

    @staticmethod
    def load(filepath):
        """Loads the entire model object from a pickle file."""
        with open(filepath, 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_model 

    def q_measure(self): 
        score = 0
        # Get the best actions for each state
        q_actions = {state: np.argmax([self.Q[(state, action)] for action in range(5)]) for state in self.states}
        for state in optimum_q_kinematics.keys():
            if q_actions[state] == optimum_q_kinematics[state]:
                score+=1
        return 100*(score/len(optimum_q_kinematics))
    
    def get_config_control(self):
        config_control = self.config.copy() 
        # Change to default values
        config_control.update({
            "duration": 120,
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "collision_reward": -1,
            "high_speed_reward": 0.4,
            "reward_speed_range": [20,30],
            "right_lane_reward": 0.1,
            "lane_change_reward": 0
        })
        return config_control

    def evaluate(self, iterations=10, render=False): 
        render_mode = 'human' if render else None
        controlled_env = gym.make('highway-fast-v0', render_mode=render_mode, config=self.get_config_control())
        evaluation_data = []
        speed_data  = []
        for i in tqdm(range(iterations)): 
            self.current_obs, info = controlled_env.reset(seed = np.random.randint(1000))
            done, steps = False, 0
            cum_reward = 0
            cum_speed = 0
            while not done:
                state = self.get_state()
                action = self.policy_Q(state)
                self.current_obs, reward, done, truncate, info = controlled_env.step(action)
                done = done | truncate
                cum_speed += self.current_obs[0,2]
                cum_reward += reward
                steps += 1
            speed_data.append(cum_speed/steps)
            evaluation_data.append((cum_reward, steps))

        mean_reward = np.mean([reward for reward, _ in evaluation_data])
        mean_steps = np.mean([steps for _, steps in evaluation_data])
        mean_reward_per_step = mean_reward/mean_steps
        mean_speed = np.mean(speed_data)
        controlled_env.close()

        return mean_reward, mean_steps, mean_reward_per_step, mean_speed
    
    def evaluate_during_train(self, current_episode, iterations=4, max_steps=200):
        controlled_env = gym.make('highway-fast-v0', config=self.get_config_control())
        reward_hist, speed_hist, actions_hist, steps_hist  = [], [], [], []
        print('Evaluating the model during training...')
        for i in range(iterations):
            obs, info = controlled_env.reset(seed = np.random.randint(100000))
            self.current_obs = obs
            done, steps = False, 0
            cum_reward = 0
            while not done and steps < max_steps:
                state = self.get_state()
                action = self.policy_Q(state)
                actions_hist.append(action)
                self.current_obs, reward, done, truncate, info = controlled_env.step(action)
                done = done | truncate
                speed_hist.append(self.current_obs[0,2])
                cum_reward += reward
                steps += 1
            steps_hist.append(steps)
            reward_hist.append(cum_reward)
        
        mean_reward = np.mean(reward_hist)
        mean_speed = np.mean(speed_hist)
        mean_steps = np.mean(steps_hist)
        actions_hist = np.array(actions_hist)
        action_distribution = np.array([np.sum(actions_hist == i) for i in range(5)]) / len(actions_hist)

        self.rewards_ev_hist.append(mean_reward)
        self.speed_ev_hist.append(mean_speed)
        self.steps_ev_hist.append(mean_steps)
        self.actions_ev_dist.append(action_distribution)

        # Now plot the results
        clear_output(wait=True)
        plt.figure(figsize=[16, 10])
        plt.suptitle(f"Stats for episode {current_episode}")

        plt.subplot(2, 2, 1)
        plt.plot(self.rewards_ev_hist, label='Rewards')
        plt.title('Mean reward per episode')
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.plot(self.speed_ev_hist, label='Speed')
        plt.title('Mean speed per episode')
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.plot(self.steps_ev_hist, label='Steps')
        plt.title('Mean steps per episode')
        plt.grid()

        plt.subplot(2, 2, 4)
        action_distribution_history = np.array(self.actions_ev_dist)
        for i in range(5):
            plt.plot(action_distribution_history[:, i], label=f'{decode_meta_action(i)}')
        plt.title('Action distribution')
        plt.grid()
        plt.legend()
        plt.show()

        controlled_env.close()


class Q_learning(Algorithm):
    def __init__(
        self,
        **kwargs,
        ):
        """
        Q-learning class constructor
        Arguments:
            alpha: float, the learning rate
            gamma: float, the discount factor
            m: int, the number of episodes to train the agent for
            epsilon: float, the epsilon value for the epsilon-greedy policy
            print_stats: bool, whether to print the statistics during initialization
        """
        super().__init__(**kwargs)
        self.algorithm_type = 'Q_learning'

    def train(self, m=100, max_steps=200, verbose=0, render=False, show_progress=25):
        render_mode = 'human' if render else None
        env = gym.make('highway-v0', render_mode=render_mode, config=self.config) 
        q_explored = np.count_nonzero(list(self.Q.values()))
        for i in tqdm(range(m)):
            # For each episode, reset the environment and get the initial state
            obs, info = env.reset(seed = np.random.randint(1000))
            self.current_obs = obs
            state = self.get_state()
            cum_reward,cum_reward_compare,steps, last_state, done = 0,0,0, None, False

            while not done and steps < max_steps:
                # Get the action to make in the current state, and step the environment
                action, _ = self.epsilon_greedy(state)
                next_obs, reward, done, truncate, info = env.step(action)
                cum_reward_compare += reward
                next_reward = fix_reward(reward, next_obs[0], action, self.to_right_reward, self.to_right_skewness, self.change_lane_reward)
                self.current_obs = next_obs
                next_state = self.get_state()

                print(f'{steps}:', state, decode_meta_action(action), next_reward) if verbose > 2 else None
                if done:
                    last_state = state

                if self.Q[(state, action)] == 0:
                    q_explored += 1
                cum_reward += next_reward
                
                self.Q[(state, action)] += self.alpha*(next_reward + self.gamma*np.max([self.Q[(next_state, a)] for a in range(5)]) - self.Q[(state,action)])
                self.Q_stats[(state, action)] += 1
                state = next_state

                # If we cant turn left or right, the car wont be turning, so dont append a turning signal
                if (state[4] == -1 and action == 0) or (state[4] == 1 and action == 2):
                    self.past_actions.append(1) 
                else:
                    self.past_actions.append(action)
                steps += 1
                print(f'Q value for {state, decode_meta_action(action)} updated from {self.Q[(state,action)]} to {self.Q[(state,action)]}') if verbose > 3 else None

            self.rewards_hist.append((cum_reward, steps))
            self.rewards_hist_compare.append((cum_reward_compare, steps))
            last_state = 'Terminal' if steps >= max_steps else last_state
            print(f"Episode {i+1} completed on state {last_state} with cumulative reward: {cum_reward}, comparable: {cum_reward_compare}") if verbose > 0 else None
            print(f"Q explored: {100*q_explored/(self.state_size*5)}. Epsilon: {self.epsilon}") if verbose > 1 else None

            # For measuring how accurate the Q function is 
            if self.special_Q: 
                q_qual = self.q_measure()
                print(f"Q measure: {q_qual} %") if verbose > 1 else None
            self.decay_epsilon(i)
            if i+1 % show_progress == 0:
                self.evaluate_during_train(iterations=3, i=i, max_steps=max_steps, render=False)
            if render: 
                env = gym.make('highway-v0', render_mode=render_mode, config=self.config) 

        env.close()


class SARSA(Algorithm):
    def __init__(
        self,
        **kwargs,
        ):
        """
        SARSA class constructor
        Arguments:
            alpha: float, the learning rate
            gamma: float, the discount factor
            m: int, the number of episodes to train the agent for
            epsilon: float, the epsilon value for the epsilon-greedy policy
            print_stats: bool, whether to print the statistics during initialization
        """
        super().__init__(**kwargs)
        self.algorithm_type = 'SARSA'

    def train(self, m=100, max_steps=200, verbose=0, render=False, show_progress=25):
        render_mode = 'human' if render else None
        env = gym.make('highway-v0', render_mode=render_mode, config=self.config) 
        q_explored = np.count_nonzero(list(self.Q.values()))
        for i in tqdm(range(m)):
            # For each episode, reset the environment and get the initial state
            obs, info = env.reset(seed = np.random.randint(1000))
            self.current_obs = obs
            state = self.get_state()
            cum_reward,cum_reward_compare,steps, last_state, done = 0,0,0, None, False

            # Get the action to make in the current state, and step the environment
            action, _ = self.epsilon_greedy(state)
            while not done and steps < max_steps:
                next_obs, reward, done, truncate, info = env.step(action)
                cum_reward_compare += reward
                next_reward = fix_reward(reward, next_obs[0], action, self.to_right_reward, self.to_right_skewness, self.change_lane_reward)
                self.current_obs = next_obs
                next_state = self.get_state()

                print(f'{steps}:', state, decode_meta_action(action), next_reward) if verbose > 2 else None
                if done:
                    last_state = state

                if self.Q[(state, action)] == 0:
                    q_explored += 1
                cum_reward += next_reward
                
                next_action, _ = self.epsilon_greedy(next_state)
                self.Q[(state, action)] += self.alpha*(next_reward + self.gamma*self.Q[(next_state, next_action)] - self.Q[(state,action)])
                self.Q_stats[(state, action)] += 1
                state = next_state
                action = next_action

                # If we cant turn left or right, the car wont be turning, so dont append a turning signal
                if (state[4] == -1 and action == 0) or (state[4] == 1 and action == 2):
                    self.past_actions.append(1) 
                else:
                    self.past_actions.append(action)
                steps += 1
                print(f'Q value for {state, decode_meta_action(action)} updated from {self.Q[(state,action)]} to {self.Q[(state,action)]}') if verbose > 3 else None

            self.rewards_hist.append((cum_reward, steps))
            self.rewards_hist_compare.append((cum_reward_compare, steps))
            last_state = 'Terminal' if steps >= max_steps else last_state
            print(f"Episode {i+1} completed on state {last_state} with cumulative reward: {cum_reward}, comparable: {cum_reward_compare}") if verbose > 0 else None
            print(f"Q explored: {100*q_explored/(self.state_size*5)}. Epsilon: {self.epsilon}") if verbose > 1 else None

            # For measuring how accurate the Q function is
            if self.special_Q:
                q_qual = self.q_measure()
                print(f"Q measure: {q_qual} %") if verbose > 1 else None
            self.decay_epsilon(i)
            
            if (i+1) % show_progress == 0:
                self.evaluate_during_train(iterations=3, current_episode=i, max_steps=max_steps)
        env.close()