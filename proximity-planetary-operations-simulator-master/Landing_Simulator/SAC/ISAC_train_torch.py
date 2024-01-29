import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gc
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym # import gym #with pip installation already imports also box2d
import math
import random
import os
import sys
import wandb
from tqdm import tqdm
 
sys.path.append('.')
sys.path.append('..')

from prioritized_replay_buffer import ReplayBuffer
from model import ValueNetwork, SoftQNetwork, PolicyNetwork

# Hyperparameters:
MAX_EPISODES = 1000
MAX_STEPS = 100
REPLAY_BUFFER_SIZE=10_000
BATCH_SIZE = 64
HIDDEN_DIM = 256
ACTION_REPEAT = 50 # Number of times to repeat each action in the 3d environment

ENV = '3d' # '2d' or '3d
WEIGHTS_FOLDER = 'ISAC_weights_'+ENV
LOAD_WEIGHTS = False
RENDER = True 
WANDB_LOG = True
WANDB_RUN_NAME = 'lander-'+ENV+'-isac'
USE_GPU_IF_AVAILABLE = False 

print('OK! All imports successful!')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if USE_GPU_IF_AVAILABLE else torch.device("cpu")
print('Device set to : ' + str(torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu' ))


def save_weights(value_net, target_value_net, soft_q_net1, soft_q_net2, policy_net):
  torch.save(value_net.state_dict(), WEIGHTS_FOLDER + '/weights_value_net.pt')
  torch.save(target_value_net.state_dict(), WEIGHTS_FOLDER + '/weights_target_value_net.pt')
  torch.save(soft_q_net1.state_dict(), WEIGHTS_FOLDER + '/weights_soft_q_net1.pt')
  torch.save(soft_q_net2.state_dict(), WEIGHTS_FOLDER + '/weights_soft_q_net2.pt')
  torch.save(policy_net.state_dict(), WEIGHTS_FOLDER + '/weights_policy_net.pt')
  

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
  #print(predicted_new_q_value.shape, log_prob.shape, predicted_value.shape, target_value_func.shape)
  
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
              name=WANDB_RUN_NAME,
              config={
                  'env': ENV,
                  'max_episodes': MAX_EPISODES,
                  'max_steps': MAX_STEPS,
                  'replay_buffer_size': REPLAY_BUFFER_SIZE,
                  'batch_size': BATCH_SIZE,
                  'hidden_dim': HIDDEN_DIM,
                  'load_weights': LOAD_WEIGHTS,
                  'device': device.type,
                  'action_repeat': ACTION_REPEAT
                }
              )
  
  if ENV == '3d':
    # from lander_gym_env import LanderGymEnv # Load simplified environment - no atmospheric disturbances
    from lander_gym_env_with_gusts import LanderGymEnv # Load environment with gusts
    env = LanderGymEnv(renders=RENDER, actionRepeat=ACTION_REPEAT)
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
  
  value_net = ValueNetwork(device, state_dim, HIDDEN_DIM).to(device)
  target_value_net = ValueNetwork(device, state_dim, HIDDEN_DIM).to(device)
  soft_q_net1 = SoftQNetwork(device, state_dim, action_dim, HIDDEN_DIM).to(device)
  soft_q_net2 = SoftQNetwork(device, state_dim, action_dim, HIDDEN_DIM).to(device)
  policy_net = PolicyNetwork(device, state_dim, action_dim, HIDDEN_DIM).to(device)
  
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
  local_buffer = [] # Since we need to push rho at the end of each episode, we cumulate the transitions and at the end we push the results to replay_buffer 
  
  frame_idx = 0
  episode = 0
  rewards = [] 
  avg_reward_list = []
  episode_reward_list = []
  steps_list = []
  
  # Training Loop:
  while episode < MAX_EPISODES:
    episode += 1
    state = env.reset()
    if ENV == '2d': state = state[0]
    episode_reward = 0
    print('\nEpisode', episode, 'starting at frame_idx = ', frame_idx)
    
    steps_done = 0
    for step in tqdm(range(MAX_STEPS)):
      if episode > 100:
        action = policy_net.get_action(state).detach()
        next_state, reward, done, *_ = env.step(action.numpy())
      else: 
        action = env.action_space.sample()
        next_state, reward, done, *_ = env.step(action)
      
      local_buffer.append( (state, action, reward, next_state, done) )
      replay_buffer.set_latest_transition(local_buffer[-1])

      state = next_state
      episode_reward += reward
      frame_idx += 1
      
      if len(replay_buffer) >= BATCH_SIZE: # update the networks
        update(BATCH_SIZE)
      
      if done:
        steps_done = step+1
        break
    
    episode_reward_list.append(episode_reward)
    steps_list.append(steps_done) # store how many steps we did in our episode
    
    # The idea is to delay the infusion of new experience to the replay_buffer 
    # avoiding overfitting so that the network will be trained more using old experience
    if episode % 4 == 0:
      print("Updating the replay buffer...")
      replay_buffer.push_transitions(local_buffer, episode_reward_list, steps_list)
      episode_reward_list = []
      local_buffer = []
      steps_list = []
    
    if episode % 10 == 0:
      save_weights(value_net, target_value_net, soft_q_net1, soft_q_net2, policy_net)
    
    if device.type == 'cuda':
      torch.cuda.empty_cache()
      gc.collect()

    rewards.append(episode_reward)
    avg_reward = np.mean(rewards[-100:])
    if WANDB_LOG: wandb.log({"episode": episode, "frame": frame_idx, "episode_reward": episode_reward, "avg_reward": avg_reward})
    print("Episode {} * Frame * {} * Episode reward {} * Avg Reward {}".format(episode, frame_idx, episode_reward, avg_reward))
    avg_reward_list.append(avg_reward)
    
  save_weights(value_net, target_value_net, soft_q_net1, soft_q_net2, policy_net)
  