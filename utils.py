import numpy as np
import random
from collections import deque
import torch
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
from torch.utils.data.dataset import IterableDataset

class MultiAgentReplayBuffer:

    def __init__(self, num_agents, max_size):
        self.max_size = max_size
        self.num_agents = num_agents
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array(reward), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        obs_batch = [[] for _ in range(
            self.num_agents)]  # [ [states of agent 1], ... ,[states of agent n] ]    ]
        indiv_action_batch = [[] for _ in range(
            self.num_agents)]  # [ [actions of agent 1], ... , [actions of agent n]]
        indiv_reward_batch = [[] for _ in range(self.num_agents)]
        next_obs_batch = [[] for _ in range(self.num_agents)]

        global_state_batch = []
        global_next_state_batch = []
        global_actions_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience

            for i in range(self.num_agents):
                obs_i = state[i]
                action_i = action[i]
                reward_i = reward[i]
                next_obs_i = next_state[i]

                obs_batch[i].append(obs_i)
                indiv_action_batch[i].append(action_i)
                indiv_reward_batch[i].append(reward_i)
                next_obs_batch[i].append(next_obs_i)

            global_state_batch.append(np.concatenate(state))
            global_actions_batch.append(torch.cat(action))
            global_next_state_batch.append(np.concatenate(next_state))
            done_batch.append(done)

        return obs_batch, indiv_action_batch, indiv_reward_batch, next_obs_batch, global_state_batch, global_actions_batch, global_next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

class RLDataset(IterableDataset):
    def __init__(self, buffer):
        self.buffer = buffer

    def __iter__(self, batch_size):
        obs_batch, indiv_action_batch, indiv_reward_batch, next_obs_batch, \
        global_state_batch, global_actions_batch, global_next_state_batch, \
        done_batch = self.buffer.sample(batch_size)

        for i in range(len(done_batch)):
            yield obs_batch[i], indiv_action_batch[i], indiv_reward_batch[i], next_obs_batch[i], \
        global_state_batch[i], global_actions_batch[i], global_next_state_batch[i], \
        done_batch[i]

def make_env(scenario_name, benchmark=False):
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env