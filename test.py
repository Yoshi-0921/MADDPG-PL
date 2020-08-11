import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import multiagent.scenarios as scenarios
from agent import DDPGAgent
from maddpg import MADDPG
from multiagent.environment import MultiAgentEnv
from utils import MultiAgentReplayBuffer, RLDataset, make_env


class MADDPG:
    def __init__(self):
        self.env = make_env(scenario_name='scenarios/new_env')#'simple_spread')
        self.num_agents = self.env.n
        self.agents = [DDPGAgent(self.env,
                                 agent_id,
                                 actor_lr=0.0,
                                 critic_lr=0.0,
                                 gamma=1.0) for agent_id in range(self.num_agents)]
        for agent in self.agents:
            #agent.actor.load_state_dict(torch.load('./saved_weights/actor_3000.weights', map_location=torch.device('cpu')))
            #agent.critic.load_state_dict(torch.load('./saved_weights/critic_3000.weights', map_location=torch.device('cpu')))
            pass
        self.reset()

    def reset(self):
        self.states = self.env.reset()
        self.step = 0

    def get_actions(self, states):
        actions = []
        for i in range(self.num_agents):
            action = self.agents[i].get_action(states[i])
            actions.append(action)
        return actions

    def run(self):
        for i in range(200):
            actions = self.get_actions(self.states)
            next_states, rewards, dones, _ = self.env.step(actions)
            self.env.render()
    
            if all(dones) or self.step == 199:# cfg.max_episode_len - 1:
                self.reset()
                break
            else:
                dones = [0 for _ in range(self.num_agents)]
                self.states = next_states
                self.step += 1

if __name__ == '__main__':
    maddpg = MADDPG()
    maddpg.run()
