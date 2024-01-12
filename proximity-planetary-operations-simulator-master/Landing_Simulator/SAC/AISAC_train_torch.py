import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.multiprocessing import Manager
import os
import fcntl
import concurrent.futures

from rwlock import RWLock
import numpy as np
import matplotlib.pyplot as plt
import gym #with pip installation already imports also box2d
import math
import random
from collections import deque
from fasteners import InterProcessLock
import setproctitle
from time import sleep

setproctitle.setproctitle("AutonomousLandingProcess")
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

async def main():
  print("The Main Process PID is: ", os.getpid())
  torch.manual_seed(42)
  torch.multiprocessing.set_start_method('spawn')
  loop = asyncio.get_running_loop()

  
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
  
  #device = torch.device("cpu")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print('Device set to : ' + str(torch.cuda.get_device_name(device)))
  async_device = torch.device("cpu")
  
  # Define Training Hyperparameters:

  replay_buffer_size = 10000
  max_steps = 500
  frame_idx = 0
  max_episodes = 2000
  episode_rewards = deque(maxlen=100)
  avg_reward_list = []
  batch_size = 128
  n_async_processes = 2

  network_hidden_dim = 256
  rwlock = RWLock()
  #weights_file_lock = InterProcessLock("weights/AISAC_weights/")#manager_shared_data.Lock() # Shared lock for file access
  network = NetworksManager(device, state_dim, action_dim, network_hidden_dim,rwlock)
  weights_filename = "weights/AISAC_weights/"
  #network.load(.weights_filename)
  network._save_sync(weights_filename)
  replay_buffer = ReplayBuffer(replay_buffer_size)
  local_buffer = [] # cumulate the transitions here and at the end of each episode push the cumulative reward (rho) to replay_buffer
  last_infusion_episode = 0
  last_plot_episode = 0
  #TODO: variaable that says if the network is updated. if yes, agents will update theirs network, otherwise its useless

      
  # Define shared stuff
  manager_shared_data = Manager()
  global_episode_counter = torch.multiprocessing.Value('i', 0)
  delayed_buffer = manager_shared_data.list()
  delayed_buffer_lock = manager_shared_data.Lock()
  
  #file_lock = 
  agents = [AsyncAgent(i, async_device, global_episode_counter, delayed_buffer, delayed_buffer_lock, network_hidden_dim, weights_filename, max_episodes, state_dim, action_dim,rwlock) for i in range(n_async_processes)]
  
  [agent.start() for agent in agents]
  print("All the ",n_async_processes," Agents are ready!")

 
  #Train with episodes:
  episode = None
  while global_episode_counter.value < max_episodes:
    state = env.reset()
    episode_reward = 0
    with global_episode_counter.get_lock():
      global_episode_counter.value += 1
      episode = global_episode_counter.value
    print(f'''Agent MAIN\tEpisode {episode} starting at frame_idx {frame_idx}''')
    step = 0
    while step <= max_steps:
      if frame_idx > 50:
        action = network.policy_net.get_action(state).detach()
        next_state, reward, done, _ = env.step(action.numpy())
      else: 
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

      local_buffer.append( [state, action, reward, next_state, done] )
      replay_buffer.set_latest_transition(local_buffer[-1])

      state = next_state
      episode_reward += reward
      frame_idx += 1
      step += 1
      if len(replay_buffer) >= batch_size:
        print("Update the network weights...", 'Replay_buffer size = ', len(replay_buffer))
        network.update(replay_buffer, batch_size)
          #await loop.run_in_executor(pool, network.save_async,weights_filename)


      
      #if global_episode_counter.value - last_plot_episode > 1000:
        #plot(frame_idx, episode_rewards)
        #last_plot_episode = global_episode_counter.value
        
      if done:
        break
      
    episode_rewards.append(episode_reward)
    avg_reward = np.mean(episode_rewards)
    avg_reward_list.append(avg_reward)
    print(f'''Agent MAIN\tEpisode {episode} FINISHED after {step} steps ==> Episode Reward: {episode_reward:3f} / Avg Reward: {avg_reward:.3f}''')
    
    for transition in local_buffer: transition.append(episode_reward) # push the cumulative reward to the replay buffer
    # maybe the push here, better to do it not every episode so that delay_local_buffer is not always taken by some agent
    # having an overall improvement in performance since we do less updates (and so less get_lock() and less waiting)
    print('1')
    with delayed_buffer_lock: 
      for transition in local_buffer: delayed_buffer.append(tuple(transition)) # push the transitions to the delay_local_buffer
    local_buffer = []
    
    # The idea is to delay the infusion  of new experience to the replay_buffer 
    # avoiding overfitting so that the network will be trained more using old experience
    if global_episode_counter.value - last_infusion_episode > 2:
      print("Main Agent is updating the shared replay buffer...")

      last_infusion_episode = global_episode_counter.value
      with delayed_buffer_lock:
        for transition in delayed_buffer: replay_buffer.push_transition(*transition)
        delayed_buffer[:] = [] # TODO check if this is the right way to clear the list
      print(f'''Replay buffer updated ==> new len: {len(replay_buffer)}''')

    # Run in a custom thread pool:
    
    #is it fine to use this inside a
    with concurrent.futures.ThreadPoolExecutor() as pool:
      print('2')
      pool.submit(network.save_async,weights_filename)
      #await loop.run_in_executor(pool,network.save_async,weights_filename)
      print('3')
      #
    #asyncio.run()
    
  [agent.terminate() for agent in agents] # delete all the agents when Main Agent finished
  delayed_buffer[:] = []
  replay_buffer.clear_buffer()
  # Delete the model to free up memory
  del network
  # Release GPU memory not in use by PyTorch
  torch.cuda.empty_cache()

  sleep(5)
  num_episodes = list(range(1,len(avg_reward_list)+1))
  print(len(avg_reward_list))
  print(avg_reward_list)
  plt.plot(num_episodes, avg_reward_list, marker='o', linestyle='-')
  plt.title('Average Reward over Episodes')
  plt.xlabel('Episodes')
  plt.ylabel('Average Reward')
  # Save the plot as a PNG file
  plt.savefig('plot.png')

    
if __name__ == '__main__':
  asyncio.run(main())