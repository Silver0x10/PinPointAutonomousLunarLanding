import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim
import numpy as np

class NetworksManager(nn.Module):
  def __init__(self, device, state_dim, action_dim, hidden_dim):
    super(NetworksManager, self).__init__()
    self.device = device

    self.value_net = ValueNetwork( self.device, state_dim, hidden_dim).to(self.device)
    self.target_value_net = ValueNetwork( self.device, state_dim, hidden_dim).to(self.device)
    self.soft_q_net1 = SoftQNetwork( self.device, state_dim, action_dim, hidden_dim).to(self.device)
    self.soft_q_net2 = SoftQNetwork( self.device, state_dim, action_dim, hidden_dim).to(self.device)
    self.policy_net = PolicyNetwork( self.device, state_dim, action_dim, hidden_dim).to(self.device)
  
    for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
      target_param.data.copy_(param.data)

    self.value_criterion = nn.MSELoss()
    self.soft_q_criterion1 = nn.MSELoss()
    self.soft_q_criterion2 = nn.MSELoss()
  
    value_lr = 3e-4
    soft_q_lr = 3e-4
    policy_lr = 3e-4

    self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
    self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
    self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
    self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
  
  def save(self, filename):
    torch.save(self.value_net.state_dict(), filename + "_value_net.pt")
    torch.save(self.target_value_net.state_dict(), filename + "_target_value_net.pt")
    torch.save(self.soft_q_net1.state_dict(), filename + "_soft_q_net1.pt")
    torch.save(self.soft_q_net2.state_dict(), filename + "_soft_q_net2.pt")
    torch.save(self.policy_net.state_dict(), filename + "_policy_net.pt")
  
  def load(self, filename):
    self.value_net.load_state_dict(torch.load(filename + "_value_net.pt"))
    self.target_value_net.load_state_dict(torch.load(filename + "_target_value_net.pt"))
    self.soft_q_net1.load_state_dict(torch.load(filename + "_soft_q_net1.pt"))
    self.soft_q_net2.load_state_dict(torch.load(filename + "_soft_q_net2.pt"))
    self.policy_net.load_state_dict(torch.load(filename + "_policy_net.pt"))
    
  def update(self, replay_buffer, batch_size, gamma=0.99, soft_tau=1e-2):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state).to(self.device)
    next_state = torch.FloatTensor(next_state).to(self.device)
    action = torch.FloatTensor(action).to(self.device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

    predicted_q_value1 = self.soft_q_net1(state, action)
    predicted_q_value2 = self.soft_q_net2(state, action)
    predicted_value = self.value_net(state)
    new_action, log_prob, epsilon, mean, log_std = self.policy_net.evaluate(state)

    #Q Function Training:
    target_value = self.target_value_net(next_state)
    target_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
    q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

    self.soft_q_optimizer1.zero_grad()
    q_value_loss1.backward()
    self.soft_q_optimizer1.step()

    self.soft_q_optimizer2.zero_grad()
    q_value_loss2.backward()
    self.soft_q_optimizer2.step()

    #Value Function Training:    
    predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))

    target_value_func = predicted_new_q_value - log_prob 
    #print(predicted_new_q_value.shape, log_prob.shape, predicted_value.shape, target_value_func.shape)

    value_loss = self.value_criterion(predicted_value, target_value_func.detach())

    self.value_optimizer.zero_grad()
    value_loss.backward()
    self.value_optimizer.step()
    #Policy Function Training
    policy_loss = (log_prob - predicted_new_q_value).mean()
    self.policy_optimizer.zero_grad()
    policy_loss.backward()
    self.policy_optimizer.step()  

    for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
      target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)


class ValueNetwork(nn.Module):
  def __init__(self, device, state_dim, hidden_dim, init_w=3e-3):
    super(ValueNetwork, self).__init__()
    
    self.device = device
    
    self.linear1 = nn.Linear(state_dim, hidden_dim)
    self.linear2 = nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = nn.Linear(hidden_dim, 1)
    
    self.linear3.weight.data.uniform_(-init_w, init_w)
    self.linear3.bias.data.uniform_(-init_w, init_w)
    
  def forward(self, state):
    x = nn.functional.relu(self.linear1(state))
    x = nn.functional.relu(self.linear2(x))
    x = self.linear3(x)
    return x
    
class SoftQNetwork(nn.Module):
  def __init__(self, device, num_inputs, num_actions, hidden_size, init_w=3e-3):
    super(SoftQNetwork, self).__init__()
    
    self.device = device
    
    self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    self.linear3 = nn.Linear(hidden_size, 1)
    
    self.linear3.weight.data.uniform_(-init_w, init_w)
    self.linear3.bias.data.uniform_(-init_w, init_w)
    
  def forward(self, state, action):
    x = torch.cat([state, action], 1)
    x = nn.functional.relu(self.linear1(x))
    x = nn.functional.relu(self.linear2(x))
    x = self.linear3(x)
    return x
    
class PolicyNetwork(nn.Module):
  def __init__(self, device, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
    super(PolicyNetwork, self).__init__()
    
    self.device = device
    
    self.log_std_min = log_std_min
    self.log_std_max = log_std_max
    
    self.linear1 = nn.Linear(num_inputs, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    
    self.mean_linear = nn.Linear(hidden_size, num_actions)
    self.mean_linear.weight.data.uniform_(-init_w, init_w)
    self.mean_linear.bias.data.uniform_(-init_w, init_w)
    
    self.log_std_linear = nn.Linear(hidden_size, num_actions)
    self.log_std_linear.weight.data.uniform_(-init_w, init_w)
    self.log_std_linear.bias.data.uniform_(-init_w, init_w)
    
  def forward(self, state):
    x = nn.functional.relu(self.linear1(state))
    x = nn.functional.relu(self.linear2(x))
    mean = self.mean_linear(x)
    log_std = self.log_std_linear(x)
    log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
    return mean, log_std
    
  def evaluate(self, state, epsilon=1e-6):
    mean, log_std = self.forward(state)
    std = log_std.exp()
    normal = Normal(0, 1)
    z = normal.sample()
    action = torch.tanh(mean + std*z.to(self.device))
    log_prob = Normal(mean, std).log_prob(mean + std*z.to(self.device)) - torch.log(1 - action.pow(2) + epsilon)
    log_prob = log_prob.sum(1, keepdim=True)
    return action, log_prob, z, mean, log_std
    
  def get_action(self, state):
    state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    mean, log_std = self.forward(state)
    std = log_std.exp()
    normal = Normal(0, 1)
    z = normal.sample().to(self.device)
    action = torch.tanh(mean + std*z)
    action = action.cpu()
    return action[0]
    