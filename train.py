from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import torch
import numpy as np
import pytorch_lightning as pl

from agent import DDPGAgent
from maddpg import MADDPG
from utils import MultiAgentReplayBuffer
from utils import make_env
from argparse import ArgumentParser
from utils import RLDataset
from torch.utils.data import DataLoader


class MADDPG(pl.LightningModule):
    def __init__(self):
        super(MADDPG, self).__init__()
        self.env = make_env(scenario_name='simple_spread')
        self.num_agents = self.env.n
        self.replay_buffer = MultiAgentReplayBuffer(self.num_agents, cfg.buffer_maxlen)
        self.agents = [DDPGAgent(self.env, i) for i in range(self.num_agents)]

    def populate(self, steps):
        for i in range(steps):
            self.self.agent.play_step(self.net, epsilon=1.0)

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        obs_batch, indiv_action_batch, indiv_reward_batch, next_obs_batch, \
        global_state_batch, global_actions_batch, global_next_state_batch, \
        done_batch = batch

        for i in range(self.num_agents):
            obs_batch_i = obs_batch[i]
            indiv_action_batch_i = indiv_action_batch[i]
            indiv_reward_batch_i = indiv_reward_batch[i]
            next_obs_batch_i = next_obs_batch[i]

            next_global_actions = []
            for agent in self.agents:
                next_obs_batch_i = torch.FloatTensor(next_obs_batch_i)
                indiv_next_action = agent.actor.forward(next_obs_batch_i)
                indiv_next_action = [agent.onehot_from_logits(indiv_next_action_j) for
                                     indiv_next_action_j in indiv_next_action]
                indiv_next_action = torch.stack(indiv_next_action)
                next_global_actions.append(indiv_next_action)
            next_global_actions = torch.cat(
                [next_actions_i for next_actions_i in next_global_actions], 1)

            self.agents[i].update(indiv_reward_batch_i, obs_batch_i, global_state_batch,
                                  global_actions_batch, global_next_state_batch,
                                  next_global_actions)
            self.agents[i].target_update()

    def train_dataloader(self):
        dataset = RLDataset(self.replay_buffer)
        dataloader = DataLoader(dataset=dataset, batch_size=cfg.batch_size)
        return dataloader

    def get_actions(self, states):
        actions = []
        for i in range(self.num_agents):
            action = self.agents[i].get_action(states[i])
            actions.append(action)
        return actions

if __name__ == '__main__':
    parser = pl.Trainer.add_argparse_args(ArgumentParser())
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--buffer_maxlen', default=1000000, type=int)
    parser.add_argument('--max_episode', default=10, type=int)
    parser.add_argument('--max_steps', default=10, type=int)

    cfg = parser.parse_args()
    trainer = pl.Trainer.from_argparse_args(cfg, fast_dev_run=True)

    maddpg = MADDPG()
    trainer.fit(maddpg)
