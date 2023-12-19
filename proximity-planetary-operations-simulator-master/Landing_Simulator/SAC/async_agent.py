import torch
from model import NetworksManager
from lander_gym_env import LanderGymEnv
#from torch.multiprocessing import Lock


class AsyncAgent(torch.multiprocessing.Process):
  def __init__(self, id, global_episode_counter,delay_local_buffer,manager_shared_var,hidden_dim ,filename,file_lock,max_episodes):
    super(AsyncAgent, self).__init__()
    self.id = id
    self.global_episode_counter = global_episode_counter
    self.local_buffer = []
    self.delay_local_buffer = delay_local_buffer
    self.manager_shared_var = manager_shared_var
    self.hidden_dim = hidden_dim
    self.batch_size = 0
    self.max_episodes = max_episodes

    self.filename_weights = filename
    # Create lock for file access
    self.file_lock = file_lock
    # self.env = LanderGymEnv(renders=False)

    self.env = LanderGymEnv(renders=False)
    print('OK! Environment of agent',self.id,' is configurated!')
    self.state_dim = self.env.observation_space.shape[0]
    self.action_dim = self.env.action_space.shape[0]

  def update(self):
    #TODO: every Sub-Agent uses the same hyperparameter of the main Agent?
    frame_idx = 0
    max_steps= 300
    local_episode = 0


    network = NetworksManager(self.state_dim, self.action_dim, self.hidden_dim, self.replay_buffer)
    self.load_model()

    while self.global_episode_counter.value < self.max_episodes:
      state = self.env.reset()
      episode_reward = 0
      with self.global_episode_counter.get_lock():
        self.global_episode_counter.value += 1
      print('Agent ', self.id,' Episode ', self.global_episode_counter.value, ' starting at frame_idx = ', frame_idx)
      local_episode+=1
      step = 0
      while step <= max_steps:
        if frame_idx > 50:
          action = network.policy_net.get_action(state).detach()
          next_state, reward, done, _ = self.env.step(action.numpy())
        else: 
          action = self.env.action_space.sample()
          next_state, reward, done, _ = self.env.step(action)

        print('Agent ', self.id," reward", reward, 'at step', step)

        self.local_buffer.append( [state, action, reward, next_state, done] )
        #self.replay_buffer.set_latest_transition(local_buffer[-1])

        state = next_state
        episode_reward += reward
        # episode_reward_list.append(episode_reward)
        frame_idx += 1
        step+=1

        if done:
          break
        
      for transition in self.local_buffer: transition.append(episode_reward) # push the cumulative reward to the replay buffer

      with self.delay_local_buffer.get_lock(): # push the transitions to the delay_local_buffer
        for transition in self.local_buffer: self.delay_local_buffer.value.append(tuple(transition))
      self.local_buffer = []

      # TODO load the weight periodically
      if local_episode %2 == 0: # when load the weights? remember that loading/saving files uses system calls that are slow
        self.load_model()



  
  def run(self):
    print('Starting agent', self.id)
    print('Agent', self.id, 'running episode', self.global_episode_counter.value)
    self.update()
    #with self.global_episode_counter.get_lock():
    #  self.global_episode_counter.value += 1
    print('Agent', self.id, 'finished')


  def load_model(self):
    with self.file_lock.get_lock():
        self.network.load_state_dict(torch.load(self.filename))