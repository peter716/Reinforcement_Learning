# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 09:13:59 2023

@author: Hp
"""

import numpy as np
import gym

# map=["SFFFHFFF", "FHFFFHFF", "FFFHFHFH", "HFFFHFFF","FFFFHHFF","HFFFFFFF","FFHHHFHF","FFFHFFFG"]
map = ["SFFF", "FHFH", "FFFH", "HFFG"]
new_map=["--------", "--------", "--------", "--------","--------","--------","--------","--------"]
env = gym.make('FrozenLake-v1', desc=map, map_name="8x8", is_slippery=False, render_mode="human")
map_size = len(map)
# states = [(i, j) for i in range(map_size) for j in range(map_size)]  # states are grid cell coordinates
actions = ["left","down", "right", "up"]  # actions are movement directions
# action_probs = {"up":0.2, "down":0.3, "right",0.3, "left":0.1 }
action_probs = [0.1,0.3,0.3,0.2]

# print(actions.index("up"))

def get_action_tag(action, actions):
    return actions[action]

def set_node_names(my_map):
    ## Function to set the grids with node names numbered from 0 to w-1
    
    #The counter is the node number
    counter = 0
    node_names = np.zeros((len(my_map),len(my_map)))
    for i in range(len(my_map)):
        for j in range(len(my_map)):
                node_names[i,j] = counter
                counter += 1
                
    return node_names

def  get_state_coords(state_names, state):
    x, y = np.where(state_names == state)
    
    return x[0], y[0]

#Function to choose the next action
def epsilon_greedy(Q, state):
    action=0
    
    if np.random.uniform(0, 1) < epsilon:  #explore
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])   #exploit
        
    # print(action)
    return action



#Function to learn the Q-value
def update(Qtable, state, state2, reward, action, action2):
    
    predict = Qtable[state, action]
    target = reward + gamma * Qtable[state2, action2]
    
    Qtable[state, action] = Qtable[state, action] + alpha * (target - predict)
    
    return Qtable

def get_next_position(grid, the_state, action):
    i, j = the_state
    if action == 3:
        if i > 0:
            return (i-1, j)
    elif action == 1:
        if i < grid.shape[0]-1:
            return (i+1, j)
    elif action == 0:
        if j > 0:
            return (i, j-1)
    elif action == 2:
        if j < grid.shape[1]-1:
            return (i, j+1)
    # If the action is not possible, return the current state
    return the_state


# def sarsa_algorithm()
def eval_greedy(Q, state):
    action = np.argmax(Q[state, :])
    return action


state_coords = set_node_names(map)

# print(state_coords)
env = env.unwrapped

nA = env.action_space.n
nS = env.observation_space.n

P = env.P
#Defining the different parameters

# epsilon = 0.1 #uncomment for epsilon value of 0.1. Takes a long time to run
epsilon = 0.9
# total_episodes = 1000000 #uncomment for epsilon value of 0.1. Takes a long time to run
total_episodes = 100000
max_steps = 1000
alpha = 0.1
gamma = 0.95

#Initializing the Q-matrix
Q = np.zeros((nS, nA))

if "S" in "".join(map):
        start_state = "".join(map).index("S")
else:
    print("Add start to map")


for episode in range(total_episodes):
    
    #Get coordinates of the start state
          
    x, y = get_state_coords(state_coords, start_state)
    
    # start_state = state_coords[x,y]
    
    state_coord = x,y
    
    #Set state as start state
    
    state = start_state
    
    #Get action of state
    action = epsilon_greedy(Q, state)
    
    
    
    
    #Take a number of steps
    # for i in range(max_steps):
    while True:
        #Get the next position based on the action taken
        # print("action",action)
        next_state_coords_x, next_state_coords_y  = get_next_position(state_coords, state_coord, action)
        
        # print(next_state_coords_x, next_state_coords_y)
        
        #Get the state for that action
        next_state = int(state_coords[next_state_coords_x, next_state_coords_y])
        
        
        #Set rewards and done
        # print("map value", map[next_state_coords_x][next_state_coords_y])
        if map[next_state_coords_x][next_state_coords_y] == "H":
            # print("hole")
            reward = 0
            next_action = False
            done = True
        elif map[next_state_coords_x][next_state_coords_y]  == "G":
            # print("goal")
            reward = 1
            next_action = False
            done = True          
        else: 
            reward = 0
            next_action = epsilon_greedy(Q, next_state)
            done = False
        
        
        #If next action exists, update the Q table
        if next_action:
            # print("next action found")
            Q = update(Q, state, next_state, reward, action, next_action)
            
        #If it doesn't, update this way
        else:
            # print("no next action")
            Q[state, action] += alpha * (reward- Q[state, action]) 
            
        # print(Q)
        state_coord = (next_state_coords_x, next_state_coords_y)
        state = next_state
        
        action = next_action
                        
        if done:
           # print("done")
           break


    
print("Q")
print(Q)
optimal = []
for i in range(Q.shape[0]):
    optimal.append(np.argmax(Q[i]))

print("optimal")
print(optimal)


observation, info = env.reset()

# #Only take the paths

if "S" in "".join(map):
    start_state = "".join(map).index("S")
else:
    print("Add start to map")
    
    
x, y = get_state_coords(state_coords, start_state)


terminated = False

state = int(state_coords[x][y])
while terminated != True:
    
    action = optimal[state]
    # print(action)

    observation, reward, terminated, truncated, info = env.step(action)
    
    x, y = get_state_coords(state_coords, observation)
    
    # print(observation)
    
    state = int(state_coords[x, y])
    
    # print(observation)
    
env.close()

