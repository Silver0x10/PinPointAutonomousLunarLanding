import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import numpy as np
import matplotlib.pyplot as plt
import gym #with pip installation already imports also box2d
import math
import random

import sys 
sys.path.append('.')
sys.path.append('..')

from prioritized_replay_buffer import ReplayBuffer
from async_agent import AsyncAgent
#from normalized_actions import NormalizedActions
from model import ValueNetwork, SoftQNetwork, PolicyNetwork

#Load simplified environment - no atmospheric disturbances:
#import lander_gym_env
#from lander_gym_env import LanderGymEnv

#Load environment with gusts:
from lander_gym_env_with_gusts import LanderGymEnv

print('OK! All imports successful!')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device set to : ' + str(torch.cuda.get_device_name(device)))

# TODO define the method for the async case
# def update(batch_size, gamma=0.99, soft_tau=1e-2):
#   state, action, reward, next_state, done = replay_buffer.sample(batch_size)
  
#   state = torch.FloatTensor(state).to(device)
#   next_state = torch.FloatTensor(next_state).to(device)
#   action = torch.FloatTensor(action).to(device)
#   reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
#   done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
  
#   predicted_q_value1 = soft_q_net1(state, action)
#   predicted_q_value2 = soft_q_net2(state, action)
#   predicted_value = value_net(state)
#   new_action, log_prob, epsilon, mean, log_std = policy_net.evaluate(state)
  
#   #Q Function Training:
#   target_value = target_value_net(next_state)
#   target_q_value = reward + (1 - done) * gamma * target_value
#   q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())
#   q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())
  
#   soft_q_optimizer1.zero_grad()
#   q_value_loss1.backward()
#   soft_q_optimizer1.step()
  
#   soft_q_optimizer2.zero_grad()
#   q_value_loss2.backward()
#   soft_q_optimizer2.step()
  
#   #Value Function Training:    
#   predicted_new_q_value = torch.min(soft_q_net1(state, new_action), soft_q_net2(state, new_action))
  
#   target_value_func = predicted_new_q_value - log_prob 
#   #print(predicted_new_q_value.shape, log_prob.shape, predicted_value.shape, target_value_func.shape)
  
#   value_loss = value_criterion(predicted_value, target_value_func.detach())
  
#   value_optimizer.zero_grad()
#   value_loss.backward()
#   value_optimizer.step()
#   #Policy Function Training
#   policy_loss = (log_prob - predicted_new_q_value).mean()
#   policy_optimizer.zero_grad()
#   policy_loss.backward()
#   policy_optimizer.step()  
  
#   for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
#     target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

def plot(frame_idx, rewards):
  plt.figure(figsize=(20,5))
  plt.subplot(131)
  plt.title('frame: %s. reward: %s' % (frame_idx, rewards[-1]))
  plt.plot(rewards)
  plt.show()    
    
# Initialize and Run executable:

def main():
  torch.manual_seed(42)
  torch.multiprocessing.set_start_method('spawn')
  global_episode_counter = torch.multiprocessing.Value('i', 0)
  
  env = LanderGymEnv(renders=False)
  #env = NormalizedActions(env)  
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
  
  value_net.share_memory()
  target_value_net.share_memory()
  soft_q_net1.share_memory()
  soft_q_net2.share_memory()
  policy_net.share_memory()
  
  for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)
    
  value_criterion = nn.MSELoss()
  soft_q_criterion1 = nn.MSELoss()
  soft_q_criterion2 = nn.MSELoss()
  
  value_lr = 3e-4
  soft_q_lr = 3e-4
  policy_lr = 3e-4
  

  # optimizer = SharedAdam(policy.actor_critic.parameters(), lr=1e-3)
  # optimizer.share_memory()
  value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
  soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)
  soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)
  policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
  
  replay_buffer_size=100000
  replay_buffer = ReplayBuffer(replay_buffer_size)
  # Since we need to push rho at the end of each episode
  # we cumulate the transitions and at the end we push the results to replay_buffer
  local_buffer = []
  
  #Define Training Hyperparameters:
  max_frames = 1000
  max_steps = 100
  frame_idx = 0
  episode = 0
  #total_episodes = 10
  rewards = [] 
  avg_reward_list = []
  batch_size = 2 # 128
  n_threads = 4

  episode_reward_list = []
  steps_list = []
  
  agents = [AsyncAgent(i, global_episode_counter) for i in range(n_threads)]

  [agent.start() for agent in agents]
  [agent.join() for agent in agents]
  
  
  
  # TODO implement in each agent for the async case
  # #Train with episodes:
  # while frame_idx < max_frames:
  #   state = env.reset()
  #   episode_reward = 0
  #   episode += 1
  #   print('\nEpisode', episode, 'starting at frame_idx = ', frame_idx)
  #   step = 0
  #   while step <= max_steps:
  #     if frame_idx > 50:
  #       action = policy_net.get_action(state).detach()
  #       next_state, reward, done, _ = env.step(action.numpy())
  #     else: 
  #       action = env.action_space.sample()
  #       next_state, reward, done, _ = env.step(action)
      
  #     print("reward", reward, 'at step', step)

  #     local_buffer.append( (state, action, reward, next_state, done) )
  #     replay_buffer.set_latest_transition(local_buffer[-1])

  #     state = next_state
  #     episode_reward += reward
  #     episode_reward_list.append(episode_reward)
  #     frame_idx += 1
  #     step+=1
      
  #     if len(replay_buffer) >= batch_size:#*2 # update the networks
  #       print("Update the network weights...")
  #       update(batch_size)
      
  #     if frame_idx % 1000 == 0:
  #       plot(frame_idx, rewards)
        
  #     if done:
  #       break

  #   #we remember how many step we did in our episode
  #   steps_list.append(step)
  #   # The idea is to delay the infusion  of new experience to the replay_buffer 
  #   # avoiding overfitting so that the network will be trained more using old experience
  #   if episode % 2 == 0:
  #     print("Updating the replay buffer...")
  #     replay_buffer.push_transitions(local_buffer, episode_reward_list, steps_list)
  #     episode_reward_list = []
  #     local_buffer = []
  #     steps_list = []

  #   rewards.append(episode_reward)
  #   avg_reward = np.mean(rewards[-100:])
  #   print("Frame * {} * Avg Reward is ==> {}".format(frame_idx, avg_reward))
  #   avg_reward_list.append(avg_reward)
    
  # torch.save(value_net.state_dict(), 'AISAC_weights/weights_value_net.pt')
  # torch.save(target_value_net.state_dict(), 'AISAC_weights/weights_target_value_net.pt')
  # torch.save(soft_q_net1.state_dict(), 'AISAC_weights/weights_soft_q_net1.pt')
  # torch.save(soft_q_net2.state_dict(), 'AISAC_weights/weights_soft_q_net2.pt')
  # torch.save(policy_net.state_dict(), 'AISAC_weights/policy_net.pt')
    
  # plt.plot(avg_reward_list)
  # plt.xlabel("Episodes")
  # plt.ylabel("Avg. Episodic Reward")
  # plt.show()
  # plt.savefig('plot.png')
    
if __name__ == '__main__':
  main()