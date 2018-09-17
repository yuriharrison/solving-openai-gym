"""Solving CartPole-v1 with Deep Reinforcement Learning (DQN)"""
import os
import random
import time

import gym
import keras
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation


class DQNAgent:
    """DQN Agent

    -- Arguments
    state_size: int, required
        model data input size

    action_size: int, required
        model data output size
        
    random_range: tuple length 2, required
        range of numbers to be used in a random action
    """
    def __init__(self, state_size, action_size, random_range):
        self.state_size = state_size
        self.action_size = action_size
        self.random_range = random_range
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def save(self, state, action, reward, next_state, done):
        """Save the data from an interaction"""
        self.memory.append([state, action, reward, next_state, done])

    def act(self, state):
        """Select an action based on a given state

        The action could be an random choice or a model prediction,
        depends on the agent stage of training.
        """
        if self.epsilon > np.random.rand():
            return random.randrange(self.random_range[0], self.random_range[1])

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def train(self, batch_size):
        """Train the model in a sample from the saved data

        -- Arguments
        batch_size: int, required
            size of the sample to be trained.
        """
        if(batch_size >= len(self.memory)):
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
                target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = DQNAgent(state_size=4, action_size=2, random_range=(0,2))
    show = True
    quantity = 500

    for episode in range(quantity):
        state = env.reset()
        state = np.reshape(state, [1, 4])
    
        for time_t in range(500):
            if show:
                env.render()
            
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            agent.save(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                print("Episode: {}/{}, score: {}".format(episode, quantity, time_t))
                break

        agent.train(32)

    env.close()
