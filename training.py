import pickle
import time # Necessary to evaluate frugality
import json # Necessary to export your results
import DiscreteEnvironment as DiscreteEnvironment # Imposed Discrete Environment
import DiscreteEnvironment_modified as DiscreteEnvironment_modified # Imposed Discrete Environment
import env
import numpy as np
import pymgrid

with open('data/building_1.pkl', 'rb') as f:
    building_1 = pickle.load(f)

with open('data/building_2.pkl', 'rb') as f:
    building_2 = pickle.load(f)
    
with open('data/building_3.pkl', 'rb') as f:
    building_3 = pickle.load(f)

buildings = [building_1, building_2, building_3]

def init_qtable(env, nb_action, building):
    """
    Initialize the Q-table of the Q-learning.
    The state is an intersection between net load possibilities (from -(pv production) to load),
    and the state of charge of the battery (from soc_min and soc_max).

    -- Input :
        env : Enivronment object
        nb_action : Number of action can be taken by each state during the Q-learning [Integer]

    -- Ouput :
        Q : A dict of state containing the value of weight of each action for each state
            (Dict of Dict)
    """

    state = []
    Q = {}

    # --- Defining state possibilities ---------------------------
    for i in range(-int(env.parameters["PV_rated_power"] - 1), int(env.parameters["load"] + 2)):
        for j in np.arange(round(env.battery.soc_min, 1), round(env.battery.soc_max + 0.1, 1), 0.1):
            state.append((i, round(j, 1)))

    # --- Initialize Q(s, a) at zero ----------------------------
    for s in state:
        Q[s] = {}
        for a in range(nb_action):
            if building==3:
                if a==4:
                    Q[s][a] = -50
                elif a==6:
                    Q[s][a] = -30
                else:
                    Q[s][a] = 0
            else:
                Q[s][a] = 0

    return Q

def epsilon_decreasing_greedy(action, epsilon, nb_action):
    """
    Adding random aleas for the choice of actions

    -- Input :
        action : integer representing the action taken [Integer]
        epsilon : share of aleas to consider (biggest espsilon is and biggest part of
                  aleas is taken [Float]
        nb_action : Number of action can be taken by each state during the Q-learning [Integer]

    -- Output :
        action : integer representing the action taken [Integer]
        randomm : binary value to consider if a aleas has been taken for action choice [Integer]
    """
    p = np.random.random()

    if p < (1 - epsilon):
        randomm = 0
        return action, randomm

    else:
        randomm = 1
        return np.random.choice(nb_action), randomm
    
def max_dict(d):
    """
    Returning the tuple (action, val) maximizing the reward in the Q-table depending of a state.
    Reward correspond to the amount paid to answer the consumption (count negatively)
    => Maximizing a negative number = Minimizing its absolute value

    -- Input :
        d : dict of action and reward associate of a state [Dict]

    -- Output :
        max_key : action corresponding to the maximal reward [Integer]
        max_value : value of the maximal reward [Integer]
    """
    max_key = None
    max_val = float("-inf")

    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k

    return max_key, max_val

def update_epsilon(epsilon):
    """
    Update epsilon value (share of aleas in the choice of an action) to minimize it through iteration.

    -- Input :
        epsilon : share of aleas to consider (biggest espsilon is and biggest part of
                  aleas is taken [Float]

    -- Output :
        epsilon (updated) : share of aleas to consider (biggest espsilon is and biggest part of
                            aleas is taken [Float]
    """
    epsilon = epsilon - epsilon * 0.02

    if epsilon < 0.1:
        epsilon = 0.1

    return epsilon

def change_name_action(idx, building):
    """
    Print function
    """
    if building==3:
        if idx == 0:
            action_name = "PV + Charge + Export"
        elif idx == 5:
            action_name = "PV + Discharge + Import"
        elif idx == 2:
            action_name = "Import"
        elif idx == 3:
            action_name = "Full Export"
        elif idx == 4:
            action_name = "Genset"
        elif idx == 1:
            action_name = "Export/Import"
        elif idx == 6:
            action_name = "Genset"
    else:
        if idx == 0:
            action_name = "PV + Charge + Export"
        elif idx == 1:
            action_name = "PV + Discharge + Import"
        elif idx == 2:
            action_name = "Import"
        elif idx == 3:
            action_name = "Full Export"
        elif idx == 4:
            action_name = "Export/Import"
    
    return action_name

def print_welcome(idx):
    """
    Print function
    """
    if idx == 0:
        print("------------------------------------")
        print("|        WELCOME TO PYMGRID        |")
        print("------------------------------------")
    elif idx == 1:

        print("t -     STATE  -  ACTION - COST")
        print("================================")

def training_Q_Learning_DE(env, nb_action, building, horizon):

    # --- Defining parameters ----------------------------------
    Q = init_qtable(env.mg, nb_action, building)
    nb_state = len(Q)
    nb_episode = 15
    alpha = 0.1    #  Learning rate
    epsilon = 0.1  #  Aleas
    gamma = 0.99
    record_cost = []
    t0 = time.time()
    t = t0
    print_training = "Training Progressing .   "
    print_welcome(0)
    print("\n")

    for e in range(nb_episode + 1):

        # --- Initialize episode variables --------------------------
        episode_cost = 0
        env.reset()
        net_load = round(env.mg.load - env.mg.pv)
        soc = round(env.mg.battery.soc, 1)
        s = (net_load, soc)  # First state
        a = max_dict(Q[s])[0]  # First action
        a, randomm = epsilon_decreasing_greedy(a, epsilon, nb_action)  # Adding aleas in the first action

        # --- Q-learning accros horizon ------------------------------
        for i in range(horizon):

            # Run action choosen precedently
            status, reward, done, info = env.step(a)
            
            # Compute cost with the previous actions
            r = reward
            episode_cost = env.get_cost()

            # Update variables depending on the precedent action
            net_load = round(env.mg.load - env.mg.pv)
            soc = round(env.mg.battery.soc, 1)
            s_ = (net_load, soc)
            a_ = max_dict(Q[s_])[0]

            if i == horizon - 1:
                Q[s][a] += alpha * (r - Q[s][a])

            # Update reward depending on the action choosen
            else:
                old_Q = Q[s][a]  # Previous reward
                target = (r + gamma * Q[s_][a_])  # Target = reward of the previous action + expectation of reward of the last action
                td_error = target - Q[s][a]  # Difference of cost between two episode
                Q[s][a] = (1 - alpha) * Q[s][a] + alpha * td_error  # Update weight in the Q-table with the reward of the last action
            s, a = s_, a_
        epsilon = update_epsilon(epsilon)

    return Q

def testing_Q_Learning_DE(env, Q, horizon, building):

    # --- Initialize variables --------------------------
    env.reset()
    net_load = round(env.mg.load - env.mg.pv)
    soc = round(env.mg.battery.soc, 1)
    s = (net_load, soc)
    a = max_dict(Q[s])[0]
    total_cost = 0
    print_welcome(1)

    # --- Q-learning accros horizon ----------------------
    for i in range(horizon):

        # Run action choosen precedently
        action_name = change_name_action(a, building)        
        status, reward, done, info = env.step(a)
        cost = - reward
        total_cost = env.get_cost()

        # Print function
        #if i % 500 == 0 or i == horizon - 1:
        #print(i, " -", (int(net_load), soc), action_name, round(total_cost, 1), "€")

        #  Update variables depending on the last action
        net_load = round(env.mg.load - env.mg.pv)
        soc = round(env.mg.battery.soc, 1)

        #  Defining the next state and action corresponding
        s_ = (net_load, soc)
        a_ = max_dict(Q[s_])[0]
        s, a = s_, a_
        
    return total_cost

train_start = time.process_time()

"""
Training code
"""
Q0_DE = training_Q_Learning_DE(building_environments[0], 5, 1, 8757)
Q1_DE = training_Q_Learning_DE(building_environments[1], 5, 2, 8757)
Q2_DE = training_Q_Learning_DE(building_environments[2], 7, 3, 8757)

train_end = time.process_time()

train_frugality = train_end - train_start
print(train_frugality)

