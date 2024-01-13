import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym # import gym
import math
import random
import os
import sys 
import wandb

sys.path.append('.')
sys.path.append('..')

from replay_buffer import ReplayBuffer
#from normalized_actions import NormalizedActions
from model import ValueNetwork, SoftQNetwork, PolicyNetwork

#Load simplified environment - no atmospheric disturbances:
#import lander_gym_env
#from lander_gym_env import LanderGymEnv

# Hyperparameters:
MAX_FRAMES = 100000
MAX_STEPS = 1000
WEIGHTS_FOLDER = 'SAC_weights'
LOAD_WEIGHTS = False
ENV = '3d' # '2d' or '3d
RENDER = False 
REPLAY_BUFFER_SIZE=100000
BATCH_SIZE = 128
WANDB_LOG = True

#Load environment with gusts:
from lander_gym_env_with_gusts import LanderGymEnv

print('OK! All imports successful!')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device set to : ' + str(torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu' ))
    
  
def update(batch_size, gamma=0.99, soft_tau=1e-2):
  state, action, reward, next_state, done = replay_buffer.sample(batch_size)
  
  state = torch.FloatTensor(state).to(device)
  next_state = torch.FloatTensor(next_state).to(device)
  action = torch.FloatTensor(action).to(device)
  reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
  done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
  
  predicted_q_value1 = soft_q_net1(state, action)
  predicted_q_value2 = soft_q_net2(state, action)
  predicted_value = value_net(state)
  new_action, log_prob, epsilon, mean, log_std = policy_net.evaluate(state)
  
  #Q Function Training:
  target_value = target_value_net(next_state)
  target_q_value = reward + (1 - done) * gamma * target_value
  q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())
  q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())
  
  soft_q_optimizer1.zero_grad()
  q_value_loss1.backward()
  soft_q_optimizer1.step()
  
  soft_q_optimizer2.zero_grad()
  q_value_loss2.backward()
  soft_q_optimizer2.step()
  
  #Value Function Training:    
  predicted_new_q_value = torch.min(soft_q_net1(state, new_action), soft_q_net2(state, new_action))
  target_value_func = predicted_new_q_value - log_prob
  value_loss = value_criterion(predicted_value, target_value_func.detach())
  
  value_optimizer.zero_grad()
  value_loss.backward()
  value_optimizer.step()
  
  #Policy Function Training
  policy_loss = (log_prob - predicted_new_q_value).mean()
  policy_optimizer.zero_grad()
  policy_loss.backward()
  policy_optimizer.step()  
  
  for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

if __name__ == '__main__':
  
  if WANDB_LOG:
    wandb.login(key='efa11006b3b5487ccfc221897831ea5ef2ff518f')
    wandb.init(project='lunar_lander', 
              name='lander-'+ENV+'-sac',
              config={
                  'env': ENV,
                  'max_frames': MAX_FRAMES,
                  'max_steps': MAX_STEPS,
                  'replay_buffer_size': REPLAY_BUFFER_SIZE,
                  'batch_size': BATCH_SIZE,
                  'load_weights': LOAD_WEIGHTS
                }
              )
  
  if ENV == '3d':
    env = LanderGymEnv(renders=RENDER)
    #env = NormalizedActions(env) 
  else:  
    render_mode = 'human' if RENDER else None
    env = gym.make("LunarLander-v2", render_mode=render_mode, continuous = True, gravity = -10.0, enable_wind = False, wind_power = 15.0, turbulence_power = 1.5)
  
  print('OK! Environment configuration successful!')
  state_dim = env.observation_space.shape[0]
  print("Size of state space -> {}".format(state_dim))
  action_dim = env.action_space.shape[0]
  print("Size of action space -> {}".format(action_dim))
  upper_bound = env.action_space.high[0]
  lower_bound = env.action_space.low[0]
  print("max value of action -> {}".format(upper_bound))
  print("min value of action -> {}".format(lower_bound))
  hidden_dim = 256
  
  value_net = ValueNetwork(device, state_dim, hidden_dim).to(device)
  target_value_net = ValueNetwork(device, state_dim, hidden_dim).to(device)
  soft_q_net1 = SoftQNetwork(device, state_dim, action_dim, hidden_dim).to(device)
  soft_q_net2 = SoftQNetwork(device, state_dim, action_dim, hidden_dim).to(device)
  policy_net = PolicyNetwork(device, state_dim, action_dim, hidden_dim).to(device)
  
  if not os.path.exists(WEIGHTS_FOLDER):
    os.makedirs(WEIGHTS_FOLDER)
  else:
    if LOAD_WEIGHTS:
      value_net.load_state_dict(torch.load(WEIGHTS_FOLDER + '/weights_value_net.pt'))
      target_value_net.load_state_dict(torch.load(WEIGHTS_FOLDER + '/weights_target_value_net.pt'))
      soft_q_net1.load_state_dict(torch.load(WEIGHTS_FOLDER + '/weights_soft_q_net1.pt'))
      soft_q_net2.load_state_dict(torch.load(WEIGHTS_FOLDER + '/weights_soft_q_net2.pt'))
      policy_net.load_state_dict(torch.load(WEIGHTS_FOLDER + '/weights_policy_net.pt'))
    else:
      for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)
    
  value_criterion = nn.MSELoss()
  soft_q_criterion1 = nn.MSELoss()
  soft_q_criterion2 = nn.MSELoss()
  
  value_lr = 3e-4
  soft_q_lr = 3e-4
  policy_lr = 3e-4
  
  value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
  soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)
  soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)
  policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
  
  replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
  
  frame_idx = 0
  rewards = [] 
  avg_reward_list = []
  episode = 0
  while frame_idx < MAX_FRAMES:
    episode += 1
    state = env.reset()
    if ENV == '2d': state = state[0]
    episode_reward = 0
    print('Starting new episode at frame_idx = ', frame_idx)
    
    for step in tqdm(range(MAX_STEPS)):
      if frame_idx > 50:
        action = policy_net.get_action(state).detach()
        next_state, reward, done, *_ = env.step(action.numpy())
      elif frame_idx == 50:
        action = env.action_space.sample()
        next_state, reward, done, *_ = env.step(action)
      else: 
        action = env.action_space.sample()
        next_state, reward, done, *_ = env.step(action)
      
      replay_buffer.push(state, action, reward, next_state, done)
    
      state = next_state
      episode_reward += reward
      frame_idx += 1
    
      if len(replay_buffer) > BATCH_SIZE:
        update(BATCH_SIZE)
        
      if episode % 10 == 0:
        torch.save(value_net.state_dict(), WEIGHTS_FOLDER + '/weights_value_net.pt')
        torch.save(target_value_net.state_dict(), WEIGHTS_FOLDER + '/weights_target_value_net.pt')
        torch.save(soft_q_net1.state_dict(), WEIGHTS_FOLDER + '/weights_soft_q_net1.pt')
        torch.save(soft_q_net2.state_dict(), WEIGHTS_FOLDER + '/weights_soft_q_net2.pt')
        torch.save(policy_net.state_dict(), WEIGHTS_FOLDER + '/weights_policy_net.pt')
          
      
      if done or episode_reward > 200:
        break
      
    rewards.append(episode_reward)
    avg_reward = np.mean(rewards[-100:])
    if WANDB_LOG: wandb.log({"episode": episode, "frame": frame_idx, "episode_reward": episode_reward, "avg_reward": avg_reward})
    print("Episode {} * Frame * {} * Episode reward {} * Avg Reward {}".format(episode, frame_idx, episode_reward, avg_reward))
    avg_reward_list.append(avg_reward)
  
  torch.save(value_net.state_dict(), WEIGHTS_FOLDER + '/weights_value_net.pt')
  torch.save(target_value_net.state_dict(), WEIGHTS_FOLDER + '/weights_target_value_net.pt')
  torch.save(soft_q_net1.state_dict(), WEIGHTS_FOLDER + '/weights_soft_q_net1.pt')
  torch.save(soft_q_net2.state_dict(), WEIGHTS_FOLDER + '/weights_soft_q_net2.pt')
  torch.save(policy_net.state_dict(), WEIGHTS_FOLDER + '/weights_policy_net.pt')
    