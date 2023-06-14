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


def value_iteration(P,states,nA, gamma =0.9, tol =0.0000001):
    map_length = len(states)
    
    nS = map_length**2
    
    # value_function = np.zeros((map_length, map_length))
    
    value_function = np.zeros(nS)
    
    # policy = np.zeros((map_length, map_length))
    
    policy = [["" for i in range(map_length)] for j in range(map_length)]
    
    # policy = np.zeros((map_length, map_length))
    
    # policy = np.zeros(nS, dtype = int)
    
    
    # error = 1
    
    while True:
        
        delta = 0
        
        # new_value_function = np.zeros((map_length, map_length))
        # new_value_function = np.zeros(nS)
        
        for s in range(nS):
            old_v = value_function[s]
            
            Qs = np.zeros(nA)
            
            for a in range(nA):
                # transitions = P[s][a]
                
                # for transition in transitions:
                _, nextS, reward, term = P[s][a][0]
                
                prob  = action_probs[a]
                x, y = get_state_coords(states, nextS)
                
                terminality_check = not term
                
                # print("state", s, "action", a, "nextS", nextS, "terminate",1*terminality_check)
                
                
                # Qs[a] += prob*(reward +gamma*value_function[x,y])
                # print(value_function[nextS])
                # print(prob*(reward +gamma*value_function[nextS] * terminality_check))
                Qs[a] += prob*(reward +gamma*value_function[nextS] * terminality_check)
                
                
            x, y = get_state_coords(states, nextS)
            
            # print(x,y)
            
            # print(Qs)
            
            value_function[s] = max(Qs)
            
            delta = max(delta, abs(old_v - value_function[s]))
            
            
        if delta < tol:
            break
                
                
            # new_value_function[x,y] = max(Qs)
        
        
    for s in range(nS):
        Qs = np.zeros(nA)
        # print(s)
        
        for a in range(nA):
            transitions = P[s][a]
            
            # print(transitions)
            
            
        # for transition in transitions:
            _, nextS, reward, term = transitions[0]
            
            #Get the probability based on the action
            prob  = action_probs[a]
            # check_terminality = not term
            #Get state coordinate
            x, y = get_state_coords(states, nextS)
            
            if term:
                policy[x][y] = 0
            terminality_check = not term
            
            Qs[a]+= prob *(reward +(gamma*value_function[nextS] * terminality_check))
            
            # Qs[a]+= prob *(reward +gamma*value_function[x,y])
            
            max_as = np.where(Qs ==Qs.max())
           
            max_as = max_as[0]
        
        action_tag = get_action_tag(max_as[0], actions)
        
        # print(s)
        
        x, y = get_state_coords(states, s)
        
        # print(policy[x][y])
        # print(x,y)
        policy[x][y] = action_tag
        
    return value_function, policy




state_coords = set_node_names(map)
# print(len(state_coords)**2)




# print(state_coords)
env = env.unwrapped

nA = env.action_space.n
nS = env.observation_space.n

P = env.P


# print(P[0][0])

value_func, policy = value_iteration(P, state_coords,nA)

print("value function")
print(value_func)
print("policy")
print(policy)

observation, info = env.reset()

#Only take the paths returned by dijkstra

if "S" in "".join(map):
    start_state = "".join(map).index("S")
else:
    print("Add start to map")
    
    
x, y = get_state_coords(state_coords, start_state)


terminated = False

while terminated != True:
    
 
    action = policy[x][y]
    # print(action)

    observation, reward, terminated, truncated, info = env.step(actions.index(action))
    
    x, y = get_state_coords(state_coords, observation)
    
    # print(observation)
    
env.close()

