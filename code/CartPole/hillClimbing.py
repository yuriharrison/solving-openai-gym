"""Solving CartPole-v1 with the Hill-Climbing method"""
import time

import numpy as np
import gym


TARGET_SCORE = 500

def run_episode(env, weight):
    state = env.reset()
    totalreward = 0
    for _ in range(TARGET_SCORE):
        env.render()
        action = 0 if np.matmul(weight, state) < 0 else 1
        state, reward, done, _ = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward


def gen_random_weights():
    return np.random.rand(4) * 2 - 1


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    best_weight = None
    best_reward = 0
    weight = gen_random_weights()
    noise_scaling = .15
    for episode in range(1000):
        weight += gen_random_weights()*noise_scaling
        reward = run_episode(env, weight)
        print("Episode {}, score: {}".format(episode, reward))
        if reward > best_reward:
            best_reward = reward
            best_weight = weight
            if reward == TARGET_SCORE:
                break
    env.close()
    