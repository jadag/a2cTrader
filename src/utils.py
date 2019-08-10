import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optimizer
from torch.distributions import Categorical
import random

from collections import namedtuple, deque


class RCNN(nn.Module):

    def __init__(self, nr_in, nr_out, nr_assets,channels = 3,add_position = True):
        super(RCNN, self).__init__()
        hidden_l = 3
        sub_net_out = 10
        self.channels = channels
        hidden_sz = nr_assets * sub_net_out
        self.lstm = nn.LSTM(nr_in,sub_net_out, hidden_l )
        hidden_middle = int(hidden_sz/2)
        self.linear1 = nn.Linear(hidden_sz,hidden_middle)
        if add_position:
            self.linear2 = nn.Linear((hidden_middle+nr_assets+1),nr_out)
        else:
            self.linear2 = nn.Linear(hidden_middle,nr_out)
        
    def forward(self, quotes,position):
        
        lstm_outs = []
        
        for quote in quotes:
            len_quote = quotes.shape[-2]
            input = quote.view(-1, len_quote, self.channels)
            out, hidden = self.lstm(input)
            out = func.tanh(out[:,-1])
            lstm_outs.append(out)
            
        conc_out = torch.cat(lstm_outs,1)
        lin_out = self.linear1(conc_out)
        lin_out = func.tanh(lin_out)
        if len(position):
            out_position = torch.cat([lin_out, position.view(1,-1)],1)
        else:
            out_position = lin_out
        lin_out = self.linear2(out_position)

        return lin_out


class QModel(nn.Module):
 
    def __init__(self, nr_inputs, nr_outputs,nr_assets):
#     def __init__(self):
        super(QModel, self).__init__()
        self.Model = RCNN(nr_inputs, nr_outputs,nr_assets)
 
    def forward(self, x):
        quotes = torch.from_numpy(x['quote']).float()
        position = torch.from_numpy(x['position']).float()
        
        x = self.Model(quotes, position)
          
        return x


class PolicyModel(nn.Module):
 
    def __init__(self, nr_inputs, nr_outputs,nr_assets):
#     def __init__(self):
        super(PolicyModel, self).__init__()
        self.Model = RCNN(nr_inputs, nr_outputs,nr_assets)
 
    def forward(self, x):
        quotes = torch.from_numpy(x['quote']).float()
        position = torch.from_numpy(x['position']).float()
        
        x = self.Model(quotes, position)
        
        x = func.softmax(x, dim=-1)
        probabilities = Categorical(x)
        return probabilities


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, field_names):
   
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names)
        self.seed = random.seed(seed)
    
    def __len__(self):
        return len(self.memory) 
    
    def add(self, *step_input):
        """Add a new experience to memory."""
        e = self.experience(*step_input)
        self.memory.append(e)
        
    def sample(self, random=True):
        """Randomly sample a batch of experiences from memory."""
        batch_sz = self.batch_size
        if len(self.memory) < self.batch_size:
            batch_sz = len(self.memory) - 1
        if random:
            experiences = random.sample(self.memory, batch_sz)
        else:
            experiences = self.memory

        return experiences  # (states, actions, rewards, next_states, dones,log_prob, values)
    
    def clear(self):
        self.memory.clear()
       
    
