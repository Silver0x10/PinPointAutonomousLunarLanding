import torch
from model import NetworksManager
from lander_gym_env_with_gusts import LanderGymEnv
#from torch.multiprocessing import Lock
import numpy as np
from collections import deque


class AsyncAgent(torch.multiprocessing.Process):
  def __init__(self, id, device, global_episode_counter, delayed_buffer, delayed_buffer_lock, hidden_dim, weights_filename, weights_file_lock, max_episodes, state_dim, action_dim):
    super(AsyncAgent, self).__init__()
    self.id = id
    self.global_episode_counter = global_episode_counter
    self.local_buffer = []
    self.delayed_buffer = delayed_buffer
    self.delayed_buffer_lock = delayed_buffer_lock
    self.hidden_dim = hidden_dim
    self.batch_size = 0
    self.max_episodes = max_episodes

    self.env = None
    self.weights_filename = weights_filename
    # Create lock for file access
    self.weights_file_lock = weights_file_lock

    self.state_dim = state_dim
    self.action_dim = action_dim
    self.network = NetworksManager(device, self.state_dim, self.action_dim, self.hidden_dim)

  def rollout(self):
    #TODO: every Sub-Agent uses the same hyperparameter of the main Agent?
    frame_idx = 0
    max_steps= 300
    episode = None
    episode_rewards = deque(maxlen=100)
    avg_reward_list = []

    while self.global_episode_counter.value < self.max_episodes:
      with self.weights_file_lock: self.network.load(self.weights_filename)
      state = self.env.reset()
      episode_reward = 0
      with self.global_episode_counter.get_lock():
        self.global_episode_counter.value += 1
        episode = self.global_episode_counter.value
      print(f'''Agent {self.id}\tEpisode {episode} starting at local frame_idx {frame_idx}''')
      step = 0
      while step <= max_steps:
        if frame_idx > 50:
          action = self.network.policy_net.get_action(state).detach()
          next_state, reward, done, _ = self.env.step(action.numpy())
        else: 
          action = self.env.action_space.sample()
          next_state, reward, done, _ = self.env.step(action)

        self.local_buffer.append( [state, action, reward, next_state, done] )

        state = next_state
        episode_reward += reward
        frame_idx += 1
        step+=1

        if done:
          break
      
      episode_rewards.append(episode_reward)
      avg_reward = np.mean(episode_rewards)
      avg_reward_list.append(avg_reward)
      print(f'''Agent {self.id}\tEpisode {episode} FINISHED after {step} steps ==> Episode Reward: {episode_reward:3f} / Avg Reward: {avg_reward:.3f}''')
      
      for transition in self.local_buffer: transition.append(episode_reward) # push the cumulative reward to the replay buffer

      with self.delayed_buffer_lock:
        for transition in self.local_buffer: self.delayed_buffer.append(tuple(transition)) # The access to the list is protected by the Manager
      self.local_buffer = []

  
  def run(self):
    print('Starting agent', self.id)
    if self.env == None: self.env = LanderGymEnv(renders=False)
    print('OK! Environment of agent', self.id, 'is configurated!')
    self.rollout()
    #with self.global_episode_counter.get_lock():
    #  self.global_episode_counter.value += 1
    print('Agent', self.id, 'finished')


  def load_model(self):
    # with self.file_lock.get_lock():
    self.network.load_state_dict(torch.load(self.filename))