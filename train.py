import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, optim
from torch.utils.data import DataLoader

import multiagent.scenarios as scenarios
from agent import DDPGAgent
from maddpg import MADDPG
from multiagent.environment import MultiAgentEnv
from utils import MultiAgentReplayBuffer, RLDataset, make_env


class MADDPG(pl.LightningModule):
    def __init__(self):
        super(MADDPG, self).__init__()
        self.env = make_env(scenario_name='simple_spread')
        self.num_agents = self.env.n
        self.replay_buffer = MultiAgentReplayBuffer(self.num_agents, cfg.buffer_maxlen)
        self.agents = [DDPGAgent(self.env,
                                 agent_id,
                                 actor_lr=cfg.actor_lr,
                                 critic_lr=cfg.critic_lr,
                                 gamma=cfg.gamma) for agent_id in range(self.num_agents)]
        self.episode_rewards = list()
        self.episode = 0
        self.episode_reward = 0
        self.populate(cfg.warm_start_steps)
        self.states = self.env.reset()
        self.reset()
        if not os.path.exists(os.path.join(os.getcwd(), 'saved_weights')):
                os.mkdir(os.path.join(os.getcwd(), 'saved_weights'))

    def populate(self, steps=1000):
        states = self.env.reset()
        for i in range(steps):
            actions = self.get_actions(states)
            next_states, rewards, dones, _ = self.env.step(actions)
            self.replay_buffer.push(states, actions, rewards, next_states, dones)
            states = next_states

    def reset(self):
        self.states = self.env.reset()
        self.step = 0
        self.episode_reward = 0

    def forward(self):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Execution phase
        if optimizer_idx == 0:
            if self.current_epoch == 3000:
                torch.save(self.agents[0].actor.state_dict(), './saved_weights/actor_3000.weights')
                torch.save(self.agents[0].critic.state_dict(), './saved_weights/critic_3000.weights')
            elif self.current_epoch == 10000:
                torch.save(self.agents[0].actor.state_dict(), './saved_weights/actor_10000.weights')
                torch.save(self.agents[0].critic.state_dict(), './saved_weights/critic_10000.weights')
            elif self.current_epoch == 30000:
                torch.save(self.agents[0].actor.state_dict(), './saved_weights/actor_30000.weights')
                torch.save(self.agents[0].critic.state_dict(), './saved_weights/critic_30000.weights')
            actions = self.get_actions(self.states)
            next_states, rewards, dones, _ = self.env.step(actions)
            self.episode_reward += np.mean(rewards)

            if all(dones) or self.step == cfg.max_episode_len - 1:
                dones = [1 for _ in range(self.num_agents)]
                self.replay_buffer.push(self.states, actions, rewards, next_states, dones)
                self.episode_rewards.append(self.episode_reward)
                print()
                print(f"global_step: {self.global_step}  |  episode: {self.episode}  |  step: {self.step}|  reward: {np.round(self.episode_reward, decimals=4)}  \n")
                self.logger.experiment.add_scalar('episode_reward', self.episode_reward, self.episode)
                self.episode += 1
                self.reset()
            else:
                dones = [0 for _ in range(self.num_agents)]
                self.replay_buffer.push(self.states, actions, rewards, next_states, dones)
                self.states = next_states
                self.step += 1

        # Training phase
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
            if self.on_gpu: next_obs_batch_i = torch.cuda.FloatTensor(next_obs_batch_i.float())
            else: next_obs_batch_i = torch.FloatTensor(next_obs_batch_i.float())

            indiv_next_action = agent.actor(next_obs_batch_i)
            indiv_next_action = [agent.onehot_from_logits(indiv_next_action_j) for
                                 indiv_next_action_j in indiv_next_action]
            indiv_next_action = torch.stack(indiv_next_action)
            next_global_actions.append(indiv_next_action)
            # Soft update of target network
            if self.global_step % cfg.sync_rate == 0:
                agent.target_update()
        next_global_actions = torch.cat([next_actions_i for next_actions_i in next_global_actions], 1)

        if self.on_gpu:
            indiv_reward_batch_i = torch.cuda.FloatTensor(indiv_reward_batch_i.float())
            indiv_reward_batch_i = indiv_reward_batch_i.view(indiv_reward_batch_i.size(0), 1)
            obs_batch_i = torch.cuda.FloatTensor(obs_batch_i.float())
            global_state_batch = torch.cuda.FloatTensor(global_state_batch.float())
            #global_actions_batch = torch.stack(global_actions_batch)
            global_next_state_batch = torch.cuda.FloatTensor(global_next_state_batch.float())
        else:
            indiv_reward_batch_i = torch.FloatTensor(indiv_reward_batch_i)
            indiv_reward_batch_i = indiv_reward_batch_i.view(indiv_reward_batch_i.size(0), 1)
            obs_batch_i = torch.FloatTensor(obs_batch_i.float())
            global_state_batch = torch.FloatTensor(global_state_batch.float())
            #global_actions_batch = torch.stack(global_actions_batch)
            global_next_state_batch = torch.FloatTensor(global_next_state_batch.float())

        if optimizer_idx % 2 == 0:
            # update critic
            curr_Q = self.agents[agent_idx].critic(global_state_batch, global_actions_batch)
            next_Q = self.agents[agent_idx].critic_target(global_next_state_batch, next_global_actions)
            estimated_Q = indiv_reward_batch_i + cfg.gamma * next_Q

            critic_loss = self.loss_function(curr_Q, estimated_Q)
            torch.nn.utils.clip_grad_norm_(self.agents[agent_idx].critic.parameters(), 0.5)

            return {'loss': critic_loss, 'log': {'train_critic_loss': critic_loss}}

        elif optimizer_idx % 2 == 1:
            policy_loss = -self.agents[agent_idx].critic(global_state_batch,
                                                                global_actions_batch).mean()
            curr_pol_out = self.agents[agent_idx].actor(obs_batch_i)
            policy_loss += -(curr_pol_out ** 2).mean() * 1e-3
            torch.nn.utils.clip_grad_norm_(self.agents[agent_idx].critic.parameters(), 0.5)

            return {'loss': policy_loss, 'log': {'train_policy_loss': policy_loss}}

    def loss_function(self, curr_Q, estimated_Q):
        criterion = nn.MSELoss()
        loss = criterion(curr_Q, estimated_Q)

        return loss

    def configure_optimizers(self):
        optim_list = list()
        for agent in self.agents:
            optim_list.extend([agent.critic_optimizer, agent.actor_optimizer])

        return optim_list

    def train_dataloader(self):
        dataset = RLDataset(self.replay_buffer, cfg.batch_size)
        dataloader = DataLoader(dataset=dataset, batch_size=cfg.batch_size, num_workers=0)
        return dataloader

    def get_actions(self, states):
        actions = []
        for i in range(self.num_agents):
            action = self.agents[i].get_action(states[i])
            actions.append(action)
        return actions


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = pl.Trainer.add_argparse_args(ArgumentParser())
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--buffer_maxlen', default=1000000, type=int)
    #parser.add_argument('--max_episode', default=2000, type=int)
    parser.add_argument('--max_episode_len', default=200, type=int)
    parser.add_argument('--warm_start_steps', default=10000, type=int)
    parser.add_argument('--actor_lr', default=1e-4, type=float)
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--sync_rate', default=8, type=int)
    parser.add_argument('--loading_weights', default=True)

    cfg = parser.parse_args()
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        name='MADDPG_logs'
    )
    checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(os.getcwd(), 'saved_checkpoints/'),
    save_top_k=3,
    verbose=False,
    monitor="loss",
    save_weights_only=True,
    )
    trainer = pl.Trainer.from_argparse_args(
        cfg,
        gpus = 1,
        #fast_dev_run=True,
        max_epochs=30001,
        profiler=True,
        logger=logger)
        #checkpoint_callback=checkpoint_callback)
        #max_steps=10)

    maddpg = MADDPG()
    trainer.fit(maddpg)
