import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from reinvented_wheels import reLU


class Actor(nn.Module):
    def __init__(self, states, actions, hidden_size_1, hidden_size_2, max_q_action):
        super(Actor, self).__init__()

        self.l1_actor = nn.Linear(states, hidden_size_1)
        self.l2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.l3_actor = nn.Linear(hidden_size_2, actions)

        self.max_q_action = max_q_action

    def forward(self, x):
        print(f"Hello, I am the actor, with states {x.shape}!")
        x = self.l1_actor(x)
        x = reLU(x)
        x = self.l2(x)
        x = reLU(x)
        x = self.l3_actor(x)
        x = torch.tanh(x)
        return x + self.max_q_action


class Critic(nn.Module):
    def __init__(self, states, actions, hidden_size_1, hidden_size_2):
        super(Critic, self).__init__()
        self.l1_critic = nn.Linear(states + actions, hidden_size_1)
        self.l2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.l3_critic = nn.Linear(hidden_size_2, 1)

    def forward(self, states, actions):
        x = self.l1_critic(torch.cat([states, actions], 1))
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        return self.l3_critic(x)


class DDPG:
    def __init__(self, states, actions, hidden_size_1, hidden_size_2, max_q_action, discount=0.99, tau=0.01):

        super(DDPG, self).__init__()
        self.actor = Actor(states, actions, hidden_size_1, hidden_size_2, max_q_action)
        self.critic = Critic(states, actions, hidden_size_1, hidden_size_2)

        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.discount = discount
        self.tau = tau

    def select_actions(self, states):
        print("Select actions now!")
        states = torch.FloatTensor(states.reshape(1, -1))
        return self.actor(states).cpu().data.numpy().flatten()

    def update_parameters(self, batch):
        """ Backpropagation critic and actor
        Code based on https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char05%20DDPG/DDPG.py
        """

        state, next_state, action, reward, done = batch

        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q_estimate = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q_estimate, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss, actor_loss

    def save(self):
        logging.info("Saving actor and critic models")
        torch.save(self.actor.state_dict(), "actor.pth")
        torch.save(self.critic.state_dict(), "critic.pth")

    def load(self):
        logging.info("Loading actor and critic models")
        self.actor.load_state_dict(torch.load("actor.pth"))
        self.critic.load_state_dict(torch.load("critic.pth"))
