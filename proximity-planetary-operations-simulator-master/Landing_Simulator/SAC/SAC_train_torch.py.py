#import torch
import numpy as np
import random
import itertools
import torch
import torch.nn.functional as F


class ReplayBuffer:
  def __init__(self, capacity):
    self.capacity = capacity
    self.buffer = []
    self.rho_buffer = []
    self.position = 0
    self.threshold = 1

 
  def push(self, local_buffer, rho):
    # Where rho is the priority score for a transition ( is the cumulative reward of an episode)

    if len(self.buffer) < self.capacity:
      self.buffer.append(None)
      self.rho_buffer.append(None)
    self.buffer[self.position] = local_buffer
    self.rho_buffer[self.position] = rho
  
    self.position = (self.position + 1) % self.capacity


  def sample(self, batch_size, num_batches):
    batch_ids = []
    batch_ids.append(random.sample(range(len(self.buffer)), num_batches))

    priority_scores_batches = torch.FloatTensor([self.rho_buffer[i] for i in batch_ids])
    batches = [self.buffer[i] for i in batch_ids]

    # Now we have the scores and we can compute cosine similarity
    cosine_similarity_value = []
    for i in range(len(num_batches)):
       for j in range(i+1, len(num_batches)):  # Avoid redundant calculations (i != j)
      # Compute cosine similarity along the last dimension (dimension 1)
          cosine_similarity_value.append
          (F.cosine_similarity(priority_scores_batches[i], priority_scores_batches[j], dim=1))

    #if this flag is true then we sample one batch from the batches  using the prioritization tecnique 
    #otherwise we get one of the batches using uniform distribution
          
    SDP_flag = True if max(cosine_similarity_value) <= self.threshold else False

    if SDP_flag:
      batch_cumulative_reward_list = np.array([batch[-1] for i,batch in range(len(batches))])
      batch_index = np.argmax(batch_cumulative_reward_list)
      #batch_index is the index of the batch with the highest cumulative reward (rho)
    else:
      # Randomly sample a batch
      random_sampled_array = np.random.choice(batches, size=1)#, replace=False)


    #print(batch)
    #print('len batch = ', len(batch))
    state, action, reward, next_state, done = map(np.stack, zip(*batch))
    return state, action, reward, next_state, done
    
  def __len__(self):
    return len(self.buffer)
