import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optimizer
from torch.distributions import Categorical
import random
import matplotlib.pyplot as plt

from utils import QModel, PolicyModel,RCNN
from collections import namedtuple


class A2C:
    
    gamma = 0.99
    log_prob = []
    memory = []
    value = None
    
    def __init__(self, input_sz, act_sz, nr_assets, save_path):
        
        self.action_sz = act_sz
        
        self.critic = QModel(input_sz, 1,nr_assets)
        self.actor = PolicyModel(input_sz, act_sz,nr_assets)
        
        self.experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done", "log_prob", "values"])
        self.critic_optim = optimizer.Adam(self.critic.parameters(), lr=0.005)
        self.actor_optim = optimizer.Adam(self.actor.parameters(), lr=0.005)
        
    def calculate_return(self, rewards):    
        disc_reward = []
        cummulative = 0

        for R in rewards[::-1]:
            cummulative = cummulative * self.gamma + R
            disc_reward.insert(0, cummulative)
        disc_reward = torch.Tensor(disc_reward)
        disc_reward = (disc_reward - disc_reward.mean()) / (disc_reward.std() + 0.000001)

        return disc_reward
    
    def compute_returns(self, values, rewards, masks, gamma=0.99):
        R = values.detach()
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * (1 - masks[step])
            returns.insert(0, R)
        return torch.cat(returns).detach()
    
    def store_state(self, state, next_state, action, reward, done):
#         print('self.log_prob ',self.log_prob,self.log_prob.detach())
        e = self.experience(state, self.action_choice, reward, next_state, done, self.log_prob, self.value)
        self.memory.append(e)
    
    def get_experience(self):
        self._A2C__memory.states
        return self._A2C__memory.states
    
    def get_action(self, state):
        self.critic.eval()
        val = self.critic(state)
        self.value = val
        self.actor.eval()
        probabilities = self.actor(state)

        action = probabilities.sample()
        self.log_prob = probabilities.log_prob(action).unsqueeze(0)
        
        
        orders =np.array([0.0]*self.action_sz)
        self.action_choice = int(action.item())
        orders[self.action_choice] = 1.0
        return orders

    def return_memory(self):
        
        experiences = self.memory       
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = [e.action for e in experiences if e is not None]
        rewards = [e.reward for e in experiences if e is not None]
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = [e.done for e in experiences if e is not None]
        log_prob = [e.log_prob for e in experiences if e is not None]
        values = torch.cat([e.values for e in experiences if e is not None])
        #Empty memory after use
        self.memory = []
        return states, actions, rewards, next_states, dones, log_prob, values
    
    def train_agent(self):

        states, actions, rewards, next_states, dones, log_probs, values = self.return_memory()   
        
        returns = -self.compute_returns(values[-1], rewards, dones)
#         plt.plot(np.array(returns))
#         plt.show(block=False)
#         plt.pause(0.1)
        advantage = returns - values.view(-1)
        self.critic.train()
        
        critic_loss = 0.5 * advantage.pow(2).mean()
        
        self.actor.train()
        log_probs = torch.cat(log_probs)
        actor_loss = (-advantage.detach() * log_probs.view(-1)).mean()
        
        self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()
        
#         print('actor_loss ', actor_loss)
        actor_loss.backward()
        critic_loss.backward()
        self.critic_optim.step()
        self.actor_optim.step()
        
    def reset(self):
        self.memory = []



class TestBot(nn.Module):
    
    memory = []
    predictions = []
    
    def __init__(self, input_sz, act_sz, nr_assets, save_path):
        super(TestBot,self).__init__()
        self.Model = RCNN(input_sz, 1,1,3,False)
        self.optim = optimizer.Adam(self.Model.parameters(),lr= 0.2)
        self.loss_func = nn.MSELoss(size_average=False)
        self.data = namedtuple("data", ["state",  "next_state"])
        self.nr_assets = nr_assets+1
    
        
    def forward(self,quotes,position):
        
        x = self.Model(quotes,position)
        
        return func.tanh(x)
        
    def train_agent(self):

        x = [e.state['quote'][0] for e in self.memory if e is not None]
        y = [e.next_state['quote'][0] for e in self.memory if e is not None]
        y = np.concatenate([y])
        self.Model.train()
        self.optim.zero_grad()
        
        x = np.array([np.concatenate([x])])
        y = np.array([y[:,-1,1]])
        plt.clf()
        plt.plot(y[0])
        
        y = torch.tensor(y)
        x = torch.from_numpy(x).float()
        out= self.forward(x,[])
        
        plt.plot(np.array(out.detach()))
        
        plt.show(block = False)
        loss = self.loss_func(out,y.view(-1,1).float())
        print('loss',loss)
        loss.backward()
        self.optim.step()
        self.memory = []

    def store_state(self, state, next_state, action, reward, done):
        e = self.data(state, next_state)
        self.memory.append(e)
         
    def get_prediction(self,x):
        self.Model.eval()
        out = self.Model(x)
        return out
    
    
    def get_action(self,x):

        fake_pred = np.array([0.0]*self.nr_assets)
        fake_pred[0] =1.0 
        self.Model.eval()
        
        quotes = torch.from_numpy(x['quote'][0]).float()
        position = torch.from_numpy(x['position']).float()
        quotes= quotes.reshape(1,quotes.shape[0],-1)
        
        out = self.forward(quotes,[])
        self.predictions.append(out)
#         print('fake_pred ',fake_pred)
        return fake_pred 
    
    def reset(self):
        self.memory = []

