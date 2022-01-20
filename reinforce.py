import sys
import math
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable

from const import *
from env import env


class Policy(nn.Module):
    def __init__(self, num_inputs, action_space):
        super().__init__()
        self.action_space = action_space
        num_outputs = action_space.n


        self.conv1 = nn.Conv2d(num_inputs,6,3)
        self.conv2 = nn.Conv2d(6,12,3)
        self.conv3 = nn.Conv2d(12,24,3)
        
        self.linear1 = nn.Linear(240, 80)
        self.linear2 = nn.Linear(80, num_outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 一层卷积
        x = F.relu(self.conv2(x))  # 两层卷积
        x = F.relu(self.conv3(x))  # 三层卷积
        x = x.view(x.size(0), -1)
        x = self.linear1(x)  # 全连接层
        x = self.linear2(x)  # 全连接层
        return F.softmax(x)

class REINFORCE:
    def __init__(self,num_inputs, action_space):
        self.action_space = action_space
        self.model = Policy( num_inputs, action_space)
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()

    def select_action(self, state):
        probs = self.model(Variable(state).cuda())       
        action = probs.multinomial().data
        prob = probs[:, action[0,0]].view(1, -1)
        log_prob = prob.log()
        entropy = - (probs*probs.log()).sum()

        return action[0], log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i])).cuda()).sum() - (0.0001*entropies[i].cuda()).sum()
        loss = loss / len(rewards)
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()




agent = REINFORCE(INPUT_CHANNEL, NUM_ACTION)
env=env()

dir = 'ckpt_reinforce' 
if not os.path.exists(dir):    
    os.mkdir(dir)

for i_episode in range(100):
    state = torch.Tensor([env.initiate()])
    entropies = []
    log_probs = []
    rewards = []
    while(True):
        action, log_prob, entropy = agent.select_action(state)
        action = action.cpu()

        next_state, reward, done  = env.step(action.numpy()[0])

        entropies.append(entropy)
        log_probs.append(log_prob)
        rewards.append(reward)
        state = torch.Tensor([next_state])

        if done:
            break

    agent.update_parameters(rewards, log_probs, entropies, args.gamma)


    if i_episode%args.ckpt_freq == 0:
        torch.save(agent.model.state_dict(), os.path.join(dir, 'reinforce-'+str(i_episode)+'.pkl'))

        print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))
	
env.close()


