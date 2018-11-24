# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 18:35:45 2017

@author: ksluck
"""

import gym
from gym import wrappers
import numpy as np
import math as mt

class Environment(object):
    
    def __init__(self):
        return
        
    def reset(self):
        return
        
    def step(self, state,action, render):
        state_t2 = []
        reward = -1
        terminal = 1        
        return state_t2, reward, terminal
        
    def getStateDimensions(self):
        return 0
        
    def getActionDimensions(self):
        return 0
        
    def getActionBounds(self):
        return [0]

        
    
class BipedalWalkerEnvironment(Environment):
    
    def __init__(self):
        self.env = gym.make('BipedalWalker-v2')
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high
        
    def reset(self):
        state = self.env.reset()
        state = np.reshape(state,(1, self.state_dim))
        return state
        
    def step(self, state,action, render):    
        #action = np.reshape(action,(1,self.action_dim))
        state_t2, reward, terminal, info = self.env.step(action)
        #print(state_t2, reward, terminal, info)
        state_t2 = np.reshape(state_t2,(1,self.state_dim))
        if render:
            self.env.render()
        return state_t2, reward, terminal
        
    def getStateDimensions(self):
        return self.state_dim
        
    def getActionDimensions(self):
        return self.action_dim
        
    def getActionBounds(self):
        return self.action_bound
    def close(self):
        self.env.close()
        
class BipedalWalkerEnvironmentActionRepeat(Environment):
    
    def __init__(self):
        self.env = gym.make('BipedalWalker-v2')
        self.state_dim = self.env.observation_space.shape[0] * 3
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high
        
        #self.env = wrappers.Monitor(self.env, '/tmp/nn3/gym')
        
    def reset(self):
        state = self.env.reset()
        state = np.reshape(state,(1, int(self.state_dim/3)))
        state = np.concatenate((state[0],state[0],state[0]))
        return np.array([state])
        
    def step(self, state,action, render):    
        #action = np.reshape(action,(1,self.action_dim))
        action_repeat_state = []
        action_repeat_reward = 0.0
        action_repeat_terminal = False
        add_reward = 1;
        valid_frames = 1;
        for i in range(3):
            state_t2, reward, terminal, info = self.env.step(action)
            state_t2 = np.reshape(state_t2,(1,int(self.state_dim/3)))
            action_repeat_state = np.concatenate((action_repeat_state,state_t2[0]))
            action_repeat_reward += add_reward * reward
            valid_frames += add_reward
            action_repeat_terminal = action_repeat_terminal or terminal
            if render:
                self.env.render()
            if terminal:
                i += 1
                while i < 3:
                    action_repeat_state = np.concatenate((action_repeat_state,state_t2[0]))
                    i+=1
                break
        action_repeat_reward = action_repeat_reward/valid_frames   
        #print(state_t2, reward, terminal, info)
        
        
        return np.array([action_repeat_state]), action_repeat_reward, action_repeat_terminal
        
    def getStateDimensions(self):
        return self.state_dim
        
    def getActionDimensions(self):
        return self.action_dim
        
    def getActionBounds(self):
        return self.action_bound
        
    def close(self):
        self.env.monitor.close()
        
 

class MountainCarActionRepeat(Environment):
    
    def __init__(self):
        self.env = gym.make('MountainCarContinuous-v0')
        self.state_dim = self.env.observation_space.shape[0] * 3
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high
        
        #self.env = gym.wrappers.Monitor(self.env, '/tmp/nn3/gym')
        
    def reset(self):
        state = self.env.reset()
        state = np.reshape(state,(1, int(self.state_dim/3)))
        state = np.concatenate((state[0],state[0],state[0]))
        return np.array([state])
        
    def step(self, state,action, render):    
        #action = np.reshape(action,(1,self.action_dim))
        action_repeat_state = []
        action_repeat_reward = 0.0
        action_repeat_terminal = False
        add_reward = 1;
        valid_frames = 1;
        #print(action)
        for i in range(3):
            state_t2, reward, terminal, info = self.env.step(action)
            state_t2 = np.reshape(state_t2,(1,int(self.state_dim/3)))
            action_repeat_state = np.concatenate((action_repeat_state,state_t2[0]))
            action_repeat_reward += add_reward * reward
            valid_frames += add_reward
            action_repeat_terminal = action_repeat_terminal or terminal
            if render:
                self.env.render()
            if terminal:
                i += 1
                while i < 3:
                    action_repeat_state = np.concatenate((action_repeat_state,state_t2[0]))
                    i+=1
                break
        action_repeat_reward = action_repeat_reward/valid_frames  
       # print(action_repeat_reward)
        #print(state_t2, reward, terminal, info)
        
        
        return np.array([action_repeat_state]), action_repeat_reward, action_repeat_terminal
        
    def getStateDimensions(self):
        return self.state_dim
        
    def getActionDimensions(self):
        return self.action_dim
        
    def getActionBounds(self):
        return self.action_bound
        
    def close(self):
        self.env.monitor.close()
        
        
class LunarLanderActionRepeat(Environment):
    
    def __init__(self):
        self.env = gym.make('LunarLander-v2')
        self.state_dim = self.env.observation_space.shape[0] * 3
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high
        
        self.env = gym.wrappers.Monitor(self.env, '/tmp/nn8/gym')
        
    def reset(self):
        state = self.env.reset()
        state = np.reshape(state,(1, int(self.state_dim/3)))
        state = np.concatenate((state[0],state[0],state[0]))
        return np.array([state])
        
    def step(self, state,action, render):    
        #action = np.reshape(action,(1,self.action_dim))
        action_repeat_state = []
        action_repeat_reward = 0.0
        action_repeat_terminal = False
        add_reward = 1;
        valid_frames = 1;
        #print(action)
        for i in range(3):
            state_t2, reward, terminal, info = self.env.step(action)
            state_t2 = np.reshape(state_t2,(1,int(self.state_dim/3)))
            action_repeat_state = np.concatenate((action_repeat_state,state_t2[0]))
            action_repeat_reward += add_reward * reward
            valid_frames += add_reward
            action_repeat_terminal = action_repeat_terminal or terminal
            if render:
                self.env.render()
            if terminal:
                i += 1
                while i < 3:
                    action_repeat_state = np.concatenate((action_repeat_state,state_t2[0]))
                    i+=1
                break
        action_repeat_reward = mt.exp(action_repeat_reward/100)  
       # print(action_repeat_reward)
        #print(state_t2, reward, terminal, info)
        
        
        return np.array([action_repeat_state]), action_repeat_reward, action_repeat_terminal
        
    def getStateDimensions(self):
        return self.state_dim
        
    def getActionDimensions(self):
        return self.action_dim
        
    def getActionBounds(self):
        return self.action_bound
        
    def close(self):
        self.env.monitor.close()
        
        
        
class PendelumActionRepeat(Environment):
    
    def __init__(self):
        self.env = gym.make('Pendulum-v0')
        self.state_dim = self.env.observation_space.shape[0] * 3
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high
        self.env.seed(42) 
        
        #self.env = gym.wrappers.Monitor(self.env, '/tmp/nn3/gym')
        
    def reset(self):
        state = self.env.reset()
        state = np.reshape(state,(1, int(self.state_dim/3)))
        state = np.concatenate((state[0],state[0],state[0]))
        return np.array([state])
        
    def step(self, state,action, render):    
        #action = np.reshape(action,(1,self.action_dim))
        action_repeat_state = []
        action_repeat_reward = 0.0
        action_repeat_terminal = False
        add_reward = 1;
        valid_frames = 1;
        #print(action)
        for i in range(3):
            state_t2, reward, terminal, info = self.env.step(action)
            state_t2 = np.reshape(state_t2,(1,int(self.state_dim/3)))
            action_repeat_state = np.concatenate((action_repeat_state,state_t2[0]))
            action_repeat_reward += add_reward * reward
            valid_frames += add_reward
            action_repeat_terminal = action_repeat_terminal or terminal
            if render:
                self.env.render()
            if terminal:
                i += 1
                while i < 3:
                    action_repeat_state = np.concatenate((action_repeat_state,state_t2[0]))
                    i+=1
                break
        action_repeat_reward = action_repeat_reward/valid_frames  
       # print(action_repeat_reward)
        #print(state_t2, reward, terminal, info)
        
        
        return np.array([action_repeat_state]), action_repeat_reward, action_repeat_terminal
        
    def getStateDimensions(self):
        return self.state_dim
        
    def getActionDimensions(self):
        return self.action_dim
        
    def getActionBounds(self):
        return self.action_bound
        
    def close(self):
        self.env.monitor.close()