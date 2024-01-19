import torch
from model import NetworksManager
from lander_gym_env_with_gusts import LanderGymEnv
import setproctitle
import gymnasium as gym

# Multiple threads are subject to the global interpreter lock (GIL), whereas multiple child processes are not subject to the GIL.
# The GIL is a programming pattern in the reference Python interpreter (e.g. CPython, the version of Python you download from python.org).
# It is a lock in the sense that it uses synchronization to ensure that only one thread of execution can execute instructions at a time within a Python process.
# This means that although we may have multiple threads in our program, only one thread can execute at a time.
# The GIL is used within each Python process, but not across processes. So this is why we are using process for each agent and thread to do only I/O bound activities.

class AsyncAgent(torch.multiprocessing.Process):
  def __init__(self, id, device, global_episode_counter, delayed_buffer, delayed_buffer_available, hidden_dim, weights_folder, max_episodes, env_type, state_dim, action_dim,rwlock):
    super(AsyncAgent, self).__init__()
    print(id)
    self.id = id
    self.rwlock = rwlock
    self.global_episode_counter = global_episode_counter
    self.local_buffer = []
    self.delayed_buffer = delayed_buffer
    self.delayed_buffer_available = delayed_buffer_available
    self.hidden_dim = hidden_dim
    self.batch_size = 0
    self.max_episodes = max_episodes
    self.env_type = env_type

    self.env = None
    self.weights_folder = weights_folder


    self.state_dim = state_dim
    self.action_dim = action_dim
    self.network = NetworksManager(device, self.state_dim, self.action_dim, self.hidden_dim,rwlock)
    print(self.id)

  def rollout(self):
    #TODO: every Sub-Agent uses the same hyperparameter of the main Agent?
    frame_idx = 0
    max_steps= 300
    local_episode = 0
    self.network.load(self.weights_folder)
    while self.global_episode_counter.value < self.max_episodes:
      if local_episode % 10 == 0:
        self.network.load(self.weights_folder)
      state = self.env.reset()
      if self.env_type == '2d': state = state[0]
      episode_reward = 0
      with self.global_episode_counter.get_lock():
        self.global_episode_counter.value += 1
        episode = self.global_episode_counter.value
      print(f'''Agent {self.id}\tEpisode {episode} starting at local frame_idx {frame_idx}''')
      step = 0
      while step <= max_steps:
        if frame_idx > 50:
          action = self.network.policy_net.get_action(state).detach()
          next_state, reward, done, *_ = self.env.step(action.numpy())
        else: 
          action = self.env.action_space.sample()
          next_state, reward, done, *_ = self.env.step(action)

        self.local_buffer.append( [state, action, reward, next_state, done] )

        state = next_state
        episode_reward += reward
        frame_idx += 1
        step+=1

        if done:
          break
      local_episode +=1
      print(f'''Agent {self.id}\tEpisode {episode} FINISHED after {step} steps ==> Episode Reward: {episode_reward:3f}''')

      for transition in self.local_buffer: transition.append(episode_reward) # push the cumulative reward to the replay buffer

      self.delayed_buffer_available.wait() 
      for transition in self.local_buffer: 
        if self.delayed_buffer.full(): self.delayed_buffer.get() # removing tht oldest transition from the queue
        self.delayed_buffer.put(tuple(transition))
          
      self.local_buffer = []
      
  
  def run(self):
    print('Starting agent', self.id)
    torch.manual_seed(42)
    torch.multiprocessing.set_sharing_strategy('file_system')
    setproctitle.setproctitle("AutonomousLandingSubProcess"+str(self.id)) # give a name to the process
    # Since we need first create the process and then "connect" it to the client of Landergym
    # We need to initialize it here ( when the process is created ) and not in the constructor

    if self.env == None:
      if self.env_type == '3d':
        self.env = LanderGymEnv(renders=False)
      else:  
        self.env = gym.make("LunarLander-v2", continuous = True, gravity = -10.0, enable_wind = False, wind_power = 15.0, turbulence_power = 1.5)
    
    print('OK! Environment of agent', self.id, 'is configured! ---------------------------------------')
    self.rollout()
    print('Agent', self.id, 'finished')

