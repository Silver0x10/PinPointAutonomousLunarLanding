import torch
from model import NetworksManager
from lander_gym_env import LanderGymEnv
#from torch.multiprocessing import Lock


class AsyncAgent(torch.multiprocessing.Process):
  def __init__(self, id, device, global_episode_counter, delay_local_buffer, hidden_dim, weights_filename, weights_file_lock, max_episodes):
    super(AsyncAgent, self).__init__()
    self.id = id
    self.global_episode_counter = global_episode_counter
    self.local_buffer = []
    self.delay_local_buffer = delay_local_buffer
    self.hidden_dim = hidden_dim
    self.batch_size = 0
    self.max_episodes = max_episodes

    self.weights_filename = weights_filename
    # Create lock for file access
    self.weights_file_lock = weights_file_lock

    self.env = LanderGymEnv(renders=False)
    print('OK! Environment of agent', self.id, 'is configurated!')
    self.state_dim = self.env.observation_space.shape[0]
    self.action_dim = self.env.action_space.shape[0]
    self.network = NetworksManager(device, self.state_dim, self.action_dim, self.hidden_dim)

  def update(self):
    #TODO: every Sub-Agent uses the same hyperparameter of the main Agent?
    frame_idx = 0
    max_steps= 300
    local_episode = 0

    self.network.load(self.weights_filename)

    while self.global_episode_counter.value < self.max_episodes:
      state = self.env.reset()
      episode_reward = 0
      with self.global_episode_counter.get_lock():
        self.global_episode_counter.value += 1
      print('Agent ', self.id,' Episode ', self.global_episode_counter.value, 'starting at frame_idx = ', frame_idx)
      local_episode+=1
      step = 0
      while step <= max_steps:
        if frame_idx > 50:
          action = self.network.policy_net.get_action(state).detach()
          next_state, reward, done, _ = self.env.step(action.numpy())
        else: 
          action = self.env.action_space.sample()
          next_state, reward, done, _ = self.env.step(action)

        print('Agent ', self.id," reward", reward, 'at step', step)

        self.local_buffer.append( [state, action, reward, next_state, done] )

        state = next_state
        episode_reward += reward
        frame_idx += 1
        step+=1

        if done:
          break
        
      for transition in self.local_buffer: transition.append(episode_reward) # push the cumulative reward to the replay buffer

      with self.delay_local_buffer.get_lock(): # push the transitions to the delay_local_buffer
        for transition in self.local_buffer: self.delay_local_buffer.value.append(tuple(transition))
      self.local_buffer = []

  
  def run(self):
    print('Starting agent', self.id)
    print('Agent', self.id, 'running episode', self.global_episode_counter.value)
    self.update()
    #with self.global_episode_counter.get_lock():
    #  self.global_episode_counter.value += 1
    print('Agent', self.id, 'finished')


  def load_model(self):
    # with self.file_lock.get_lock():
    self.network.load_state_dict(torch.load(self.filename))