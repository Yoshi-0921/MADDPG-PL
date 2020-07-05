from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import torch
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from agent import DDPGAgent
from maddpg import MADDPG
from utils import MultiAgentReplayBuffer
from utils import make_env
from argparse import ArgumentParser
from utils import RLDataset
from torch.utils.data import DataLoader
from torch import optim


class MADDPG(pl.LightningModule):
    def __init__(self):
        super(MADDPG, self).__init__()
        self.env = make_env(scenario_name='simple_spread')
        self.num_agents = self.env.n
        self.replay_buffer = MultiAgentReplayBuffer(self.num_agents, cfg.buffer_maxlen)
        self.agents = [DDPGAgent(self.env, i) for i in range(self.num_agents)]
        self.episode_rewards = list()
        self.populate(cfg.warm_start_steps)

    def populate(self, steps=1000):
        for i in range(steps):
            self.self.agent.play_step(self.net, epsilon=1.0)

    def forward(self):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        obs_batch, indiv_action_batch, indiv_reward_batch, next_obs_batch, \
        global_state_batch, global_actions_batch, global_next_state_batch, \
        done_batch = batch

        agent_idx = optimizer_idx // 2

        obs_batch_i = obs_batch[agent_idx]
        indiv_action_batch_i = indiv_action_batch[agent_idx]
        indiv_reward_batch_i = indiv_reward_batch[agent_idx]
        next_obs_batch_i = next_obs_batch[agent_idx]
        next_global_actions = list()

        for agent in self.agents:
            next_obs_batch_i = torch.FloatTensor(next_obs_batch_i)
            indiv_next_action = agent.actor.forward(next_obs_batch_i)
            indiv_next_action = [agent.onehot_from_logits(indiv_next_action_j) for indiv_next_action_j in indiv_next_action]
            indiv_next_action = torch.stack(indiv_next_action)
            next_global_actions.append(indiv_next_action)
            agent.target_update()
        next_global_actions = torch.cat([next_actions_i for next_actions_i in next_global_actions], 1)

        indiv_reward_batch_i = torch.FloatTensor(indiv_reward_batch_i)
        indiv_reward_batch_i = indiv_reward_batch_i.view(indiv_reward_batch_i.size(0), 1)
        obs_batch_i = torch.FloatTensor(obs_batch_i)
        global_state_batch = torch.FloatTensor(global_state_batch)
        global_actions_batch = torch.stack(global_actions_batch)
        global_next_state_batch = torch.FloatTensor(indiv_reward_batch_i)

        if optimizer_idx % 2 == 0:
            # update critic
            curr_Q = self.agent[agent_idx].critic.forward(global_state_batch, global_actions_batch)
            next_Q = self.agent[agent_idx].critic_target.forward(global_next_state_batch, next_global_actions)
            estimated_Q = indiv_reward_batch_i + self.gamma * next_Q

            critic_loss = self.MSELoss(curr_Q, estimated_Q.detach())
            torch.nn.utils.clip_grad_norm_(self.agent[agent_idx].critic.parameters(), 0.5)

            return {'loss': critic_loss}

        elif optimizer_idx % 2 == 1:
            policy_loss = -self.agent[agent_idx].critic.forward(global_state_batch, global_actions_batch).mean()
            curr_pol_out = self.agent[agent_idx].actor.forward(obs_batch_i)
            policy_loss += -(curr_pol_out ** 2).mean() * 1e-3
            torch.nn.utils.clip_grad_norm_(self.agent[agent_idx].critic.parameters(), 0.5)

            return {'loss': policy_loss}

    def configure_optimizers(self):
        optim_list = list()
        critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        for i in range(self.num_agents):
            optim_list.extend([critic_optimizer, actor_optimizer])

        return optim_list

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
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=1,
        name='MADDPG_logs'
    )
    trainer = pl.Trainer.from_argparse_args(cfg,
                                            fast_dev_run=True,
                                            profiler=True,
                                            logger=logger)

    maddpg = MADDPG()
    trainer.fit(maddpg)
