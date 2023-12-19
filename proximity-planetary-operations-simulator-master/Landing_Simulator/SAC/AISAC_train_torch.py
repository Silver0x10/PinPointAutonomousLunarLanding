import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.multiprocessing import Manager

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
from model import NetworksManager
#from normalized_actions import NormalizedActions

#Load simplified environment - no atmospheric disturbances:
#from lander_gym_env import LanderGymEnv

#Load environment with gusts:
from lander_gym_env_with_gusts import LanderGymEnv

print('OK! All imports successful!')

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
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print('Device set to : ' + str(torch.cuda.get_device_name(device)))
  async_device = torch.device("cpu")
  
  # Define Training Hyperparameters:
  replay_buffer_size=100000
  max_steps = 100
  frame_idx = 0
  max_episodes = 10
  rewards = [] 
  avg_reward_list = []
  batch_size = 2 # 128
  n_async_processes = 1
  network_hidden_dim = 256
  
  network = NetworksManager(device, state_dim, action_dim, network_hidden_dim)
  weights_filename = "weights/AISAC_weights"
  network.save(weights_filename)
  replay_buffer = ReplayBuffer(replay_buffer_size)
  local_buffer = [] # cumulate the transitions here and at the end of each episode push the cumulative reward (rho) to replay_buffer
  last_infusion_episode = 0
  last_plot_episode = 0
  
  # Define shared stuff
  manager_shared_var = Manager()
  global_episode_counter = torch.multiprocessing.Value('i', 0)
  delay_local_buffer = manager_shared_var.list()
  weights_file_lock = manager_shared_var.Lock() # Shared lock for file access

  agents = [AsyncAgent(i, async_device, global_episode_counter, delay_local_buffer, network_hidden_dim, weights_filename, weights_file_lock, max_episodes) for i in range(n_async_processes)]
  
  [agent.start() for agent in agents]
  print("All the ",n_async_processes," Agents are ready!")
  [agent.join() for agent in agents]

  #Train with episodes:
  while global_episode_counter.value < max_episodes:
    state = env.reset()
    episode_reward = 0
    with global_episode_counter.get_lock():
      global_episode_counter.value += 1
    print('\nEpisode', global_episode_counter.value, 'starting at frame_idx = ', frame_idx)
    step = 0
    while step <= max_steps:
      if frame_idx > 50:
        action = network.policy_net.get_action(state).detach()
        next_state, reward, done, _ = env.step(action.numpy())
      else: 
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
      
      print("reward", reward, 'at step', step)

      local_buffer.append( [state, action, reward, next_state, done] )
      replay_buffer.set_latest_transition(local_buffer[-1])

      state = next_state
      episode_reward += reward
      frame_idx += 1
      step+=1
      
      if len(replay_buffer) >= batch_size:#*2 # update the networks
        print("Update the network weights...")
        network.update(replay_buffer, batch_size)
        # TODO save the weight periodically
      
      if global_episode_counter.value - last_plot_episode > 1000:
        plot(frame_idx, rewards)
        last_plot_episode = global_episode_counter.value
        
      if done:
        break
    
    for transition in local_buffer: transition.append(episode_reward) # push the cumulative reward to the replay buffer
    # maybe the push here, better to do it not every episode so that delay_local_buffer is not always taken by some agent
    # having an overall improvement in performance since we do less updates (and so less get_lock() and less waiting)

    with delay_local_buffer.get_lock(): # push the transitions to the delay_local_buffer
      for transition in local_buffer: delay_local_buffer.append(tuple(transition))
    local_buffer = []
    
    rewards.append(episode_reward)
    avg_reward = np.mean(rewards[-100:])
    print("Main Agent -> Frame * {} * Avg Reward is ==> {}".format(frame_idx, avg_reward))
    avg_reward_list.append(avg_reward)
    
    # The idea is to delay the infusion  of new experience to the replay_buffer 
    # avoiding overfitting so that the network will be trained more using old experience
    if global_episode_counter.value - last_infusion_episode > 2:
      print("Main Agent is updating the shared replay buffer...")
      last_infusion_episode = global_episode_counter.value
      with delay_local_buffer.get_lock():
        for transition in delay_local_buffer: replay_buffer.push_transition(transition)
        delay_local_buffer.clear()
        
  torch.save(network.state_dict(), 'AISAC_weights/weights.pt')
      
  plt.plot(avg_reward_list)
  plt.xlabel("Episodes")
  plt.ylabel("Avg. Episodic Reward")
  plt.show()
  plt.savefig('plot.png')
    
if __name__ == '__main__':
  main()