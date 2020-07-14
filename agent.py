import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np

from models import CentralizedCritic, Actor


class DDPGAgent:

    def __init__(self, env, agent_id, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=1e-2):
        self.env = env
        self.agent_id = agent_id
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau

        self.device = "cpu"
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.device = "cuda:0"

        self.obs_dim = self.env.observation_space[agent_id].shape[0]
        self.action_dim = self.env.action_space[agent_id].n
        self.num_agents = self.env.n

        self.critic_input_dim = int(
            np.sum([env.observation_space[agent].shape[0] for agent in range(env.n)]))
        self.actor_input_dim = self.obs_dim

        self.critic = CentralizedCritic(self.critic_input_dim, self.action_dim * self.num_agents).to(self.device)
        self.critic_target = CentralizedCritic(self.critic_input_dim,
                                               self.action_dim * self.num_agents).to(self.device)
        self.actor = Actor(self.actor_input_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.actor_input_dim, self.action_dim).to(self.device)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.MSELoss = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

    def get_action(self, state):
        state = autograd.Variable(torch.from_numpy(state).float().squeeze(0)).to(self.device)
        action = self.actor(state)
        action = self.onehot_from_logits(action)

        return action

    def onehot_from_logits(self, logits, eps=0.0):
        # get best (according to current policy) actions in one-hot form
        argmax_acs = (logits == logits.max(0, keepdim=True)[0]).float()
        if eps == 0.0:
            return argmax_acs
        # get random actions in one-hot form
        rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
            range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
        # chooses between best and random actions using epsilon greedy
        return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                            enumerate(torch.rand(logits.shape[0]))])


    def target_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
