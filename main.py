from env import env
import queue
import numpy as np
from autoEncoder import model
from reinforce import REINFORCE
from const import *
import os

env=env()
model=model()
agent = REINFORCE(INPUT_CHANNEL, NUM_ACTION)
lossBuffer=queue()
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

    agent.update_parameters(rewards, log_probs, entropies, GAMMA)


    if i_episode%1000 == 0:
        torch.save(agent.model.state_dict(), os.path.join(dir, 'reinforce-'+str(i_episode)+'.pkl'))

        print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))


model.train_epoch(env.buffer)



