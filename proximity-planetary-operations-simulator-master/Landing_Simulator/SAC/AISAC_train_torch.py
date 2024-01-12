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
import gymnasium as gym
import math
import random
from collections import deque
from fasteners import InterProcessLock
import setproctitle
from time import sleep
import sys 
import os
import wandb

# Hyperparameters:
MAX_FRAMES = 10000
MAX_STEPS = 500
MAX_EPISODES = 2000
WEIGHTS_FOLDER = 'AISAC_weights'
LOAD_WEIGHTS = False
ENV = '2d' # '2d' or '3d
RENDER = False 
REPLAY_BUFFER_SIZE=100000
BATCH_SIZE = 128
WANDB_LOG = True

setproctitle.setproctitle("AutonomousLandingProcess")
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
  
  if WANDB_LOG:
    wandb.login(key='efa11006b3b5487ccfc221897831ea5ef2ff518f')
    wandb.init(project='lunar_lander', 
              name='lander-'+ENV+'-aisac',
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
  
  #device = torch.device("cpu")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print('Device set to : ' + str(torch.cuda.get_device_name(device)))
  async_device = torch.device("cpu")
  
  # Define Training Hyperparameters:

  frame_idx = 0
  
  episode_rewards = deque(maxlen=100)
  avg_reward_list = []
  batch_size = 128
  n_async_processes = 2

  network_hidden_dim = 256
  rwlock = RWLock()
  #weights_file_lock = InterProcessLock("weights/AISAC_weights/")#manager_shared_data.Lock() # Shared lock for file access
  network = NetworksManager(device, state_dim, action_dim, network_hidden_dim,rwlock)
  
  if not os.path.exists(WEIGHTS_FOLDER):
    os.makedirs(WEIGHTS_FOLDER)
  if LOAD_WEIGHTS: network.load(WEIGHTS_FOLDER)
  network._save_sync(WEIGHTS_FOLDER)
  
  replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
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
  agents = [AsyncAgent(i, async_device, global_episode_counter, delayed_buffer, delayed_buffer_lock, network_hidden_dim, WEIGHTS_FOLDER, MAX_EPISODES, ENV, state_dim, action_dim, rwlock) for i in range(n_async_processes)]
  
  [agent.start() for agent in agents]
  print("All the ",n_async_processes," Agents are ready!")

 
  #Train with episodes:
  episode = None
  while global_episode_counter.value < MAX_EPISODES:
    state = env.reset()[0]
    episode_reward = 0
    with global_episode_counter.get_lock():
      global_episode_counter.value += 1
      episode = global_episode_counter.value
    print(f'''Agent MAIN\tEpisode {episode} starting at frame_idx {frame_idx}''')
    step = 0
    while step <= MAX_STEPS:
      if frame_idx > 50:
        action = network.policy_net.get_action(state).detach()
        next_state, reward, done, *_ = env.step(action.numpy())
      else: 
        action = env.action_space.sample()
        next_state, reward, done, *_ = env.step(action)

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
      
      if done:
        break
      
    episode_rewards.append(episode_reward)
    avg_reward = np.mean(episode_rewards)
    avg_reward_list.append(avg_reward)
    if WANDB_LOG: wandb.log({"episode": episode, "frame": frame_idx, "episode_reward": episode_reward, "avg_reward": avg_reward})
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
      pool.submit(network.save_async, WEIGHTS_FOLDER)
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