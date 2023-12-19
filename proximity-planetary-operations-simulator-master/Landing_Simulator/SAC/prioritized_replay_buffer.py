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
    self.num_batches = 2
    

  def push_transitions(self, transitions, rho_list, max_step):
    # in rho list we have the cumulative reward for each episode
    # so we will use max_step that says to us given a cumulative reward,
    # how many transitions has the same cumulative reward
    #print(len(transitions))
    #print( len(rho_list))
    assert sum(max_step) == len(rho_list), "Error in the length of rho_list"

    idx_reward = 0
    counter_items = 0

    for (state, action, reward, next_state, done) in transitions:
      if max_step[idx_reward] == counter_items:
        counter_items = 0
        idx_reward+=1
        # same rho value for every transition of the corresponding episode
      self.push_transition(state, action, reward, next_state, done, rho_list[idx_reward])
      counter_items+=1



  def push_transition(self, state, action, reward, next_state, done, rho):
    # Where rho is the priority score for a transition ( is the cumulative reward of an episode )

    if len(self.buffer) < self.capacity:
      self.buffer.append(None)
    self.buffer[self.position] = (state, action, reward, next_state, done, rho)
    self.position = (self.position + 1) % self.capacity
    #TODO: when self.position = 0, save space by imposing the buffer = none?



  def sample(self, batch_size):
    batches = []
    for i in range(self.num_batches):
      batches.append(random.sample(self.buffer, batch_size))

    priority_scores_batches = []
    for i,batch in enumerate(batches):
      priority_scores_batches.append([])
      
      for (state, action, reward, next_state, done, rho) in batch:
        priority_scores_batches[i].append(rho)

    # Now we have the scores and we can compute cosine similarity
    priority_scores_batches = torch.FloatTensor(priority_scores_batches)
    #print(priority_scores_batches)
    #print(priority_scores_batches.shape)

    cosine_similarity_value = []
    for i in range(self.num_batches):
       for j in range(i+1, self.num_batches):  # Avoid redundant calculations (i != j)
      # Compute cosine similarity along the last dimension (dimension 1)
          cosine_similarity_value.append(F.cosine_similarity(
            priority_scores_batches[i], priority_scores_batches[j], dim=0))

    #if this flag is true then we sample one batch from the batches  using the prioritization tecnique 
    #otherwise we get one of the batches using uniform distribution
    #print(cosine_similarity_value)
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
      # Get the first k elements from the sorted list (batch highest cumulative reward )
      #print('1')
      final_batch = sorted_batches[:batch_size]
    else:
      # Randomly sample a batch
      final_batch = random.choice(batches) #np.array(random.sample(batches,1)).squeeze(0)# np.random.choice(batches, size=1)#, replace=False)

    #TODO 1: now we add the latest experience on policy (latest transition) to the batch (MO/O step)
    # Select a random index
    #print(final_batch)
    random_index = random.choice(range(len(final_batch))) # random.sample(final_batch,1)
    #print(np.array(final_batch).shape)
    #print(final_batch[0])
    #print(np.array(final_batch[0]).shape)
    #print(self.latest_transition_on_policy)
    #print(np.array(self.latest_transition_on_policy).shape)
    final_batch[random_index] = self.latest_transition_on_policy
    #print(final_batch)
    state, action, reward, next_state, done, rho = map(np.stack, zip(*final_batch))
    return state, action, reward, next_state, done
    
    
  def set_latest_transition(self, transition):
    # Create a new tuple by concatenating the existing tuple with the new element
    # where None is the cumulative reward
    self.latest_transition_on_policy = tuple(transition) + (None,)

  def __len__(self):
    return len(self.buffer)
