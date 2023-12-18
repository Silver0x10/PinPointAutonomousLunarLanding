import torch

class AsyncAgent(torch.multiprocessing.Process):
  def __init__(self, id, global_episode_counter):
    super(AsyncAgent, self).__init__()
    self.id = id
    self.global_episode_counter = global_episode_counter
    self.local_buffer = []
    # self.env = LanderGymEnv(renders=False)
    print('Starting agent', self.id)
  
  def update(self):
    pass
  
  def run(self):
    print('Starting agent', self.id)
    print('Agent', self.id, 'running episode', self.global_episode_counter.value)
    self.update()
    with self.global_episode_counter.get_lock():
      self.global_episode_counter.value += 1
    print('Agent', self.id, 'finished')