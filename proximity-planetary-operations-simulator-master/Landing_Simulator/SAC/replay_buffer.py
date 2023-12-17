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
    #TODO 1: use a new variable for on-policy transition?
    self.latest_transition_on_policy = ()
    self.position = 0
    self.threshold = 0.5
    

  def push_transitions(self, transitions, rho_list, max_step):
    # in rho list we have the cumulative reward for each episode
    # so we will use max_step that says to us given a cumulative reward,
    # how many transitions has the same cumulative reward
    assert len(transitions)/max_step == rho_list, "Error in the length of rho_list"

    idx = 0
    for (state, action, reward, next_state, done) in transitions:
      if max_step == idx:
        idx = 0
      self.push_transition(state, action, reward, next_state, done, rho_list[idx])
      idx+=1



  def push_transition(self, state, action, reward, next_state, done, rho):
    # Where rho is the priority score for a transition ( is the cumulative reward of an episode )

    if len(self.buffer) < self.capacity:
      self.buffer.append(None)
    self.buffer[self.position] = (state, action, reward, next_state, done, rho)
    self.position = (self.position + 1) % self.capacity
    #TODO: when self.position = 0, save space by imposing the buffer = none?



  def sample(self, batch_size, num_batches):
    batches = []
    for i in range(num_batches):
      batches.append(random.sample(self.buffer, batch_size))

    priority_scores_batches = []
    for i,batch in enumerate(batches):
      priority_scores_batches.append(None) # TODO why none?
      # priority = batch[0][-1]
      for (state, action, reward, next_state, done, rho) in batch:
        priority_scores_batches[i].append(rho)

    # Now we have the scores and we can compute cosine similarity
    priority_scores_batches = torch.FloatTensor(priority_scores_batches)

    cosine_similarity_value = []
    for i in range(num_batches):
       for j in range(i+1, num_batches):  # Avoid redundant calculations (i != j)
      # Compute cosine similarity along the last dimension (dimension 1)
          cosine_similarity_value.append
          (F.cosine_similarity(priority_scores_batches[i], priority_scores_batches[j], dim=1))

    #if this flag is true then we sample one batch from the batches  using the prioritization tecnique 
    #otherwise we get one of the batches using uniform distribution
          
    SDP_flag = True if max(cosine_similarity_value) <= self.threshold else False

    if SDP_flag:
    # Combine and flatten the arrays using a for loop
      merged_batches = []
      for batch in batches:
          merged_batches.extend([transition for transition in batch])

      batches,priority_scores_batches,cosine_similarity_value = [],[],[] # save memory

      # Sort the list based on the last element of each tuple (rho)
      # in batch[-1] we have rho
      sorted_batches = sorted(merged_batches, key=lambda batch: batch[-1], reverse=True)
      # Get the first k elements from the sorted list
      final_batch = sorted_batches[:batch_size]

      #batch_index is the index of the batch with the highest cumulative reward (rho)
    else:
      # Randomly sample a batch
      final_batch =random.sample(batches,1)# np.random.choice(batches, size=1)#, replace=False)

    #TODO 1: now we add the latest experience on policy (latest transition) to the batch (MO/O step)
    
    state, action, reward, next_state, done = map(np.stack, zip(*final_batch))
    return state, action, reward, next_state, done
    
    


  def __len__(self):
    return len(self.buffer)
