import torch
from tqdm import tqdm
import gymnasium as gym
import numpy as np
import sys

# sys.path.append('proximity-planetary-operations-simulator-master/Landing_Simulator/SAC')
# sys.path.append('proximity-planetary-operations-simulator-master/Landing_Simulator/pybullet_data')
# sys.path.append('proximity-planetary-operations-simulator-master/Landing_Simulator/')
sys.path.append('.')
sys.path.append('..')
from model import NetworksManager

ENV = '3d' # '2d' or '3d
WEIGHTS_FOLDER = '../../../weights/AISAC_weights_3d'
# WEIGHTS_FOLDER = '../../../weights/AISAC_weights_2d_final_really'
TEST_EPISODES = 10

USE_GPU_IF_AVAILABLE = True 
MAX_STEPS = 300 if ENV == '2d' else 100
BATCH_SIZE = 64
HIDDEN_DIM = 256
RENDER = True 
ACTION_REPEAT = 50 # Number of times to repeat each action in the 3d environment


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if USE_GPU_IF_AVAILABLE else torch.device("cpu")
    
    if ENV == '3d':
        from lander_gym_env_with_gusts import LanderGymEnv
        env = LanderGymEnv(renders=RENDER, actionRepeat=ACTION_REPEAT)
    else:  
        render_mode = 'human' if RENDER else None
        env = gym.make("LunarLander-v2", render_mode=render_mode, continuous = True, gravity = -10.0, enable_wind = False, wind_power = 15.0, turbulence_power = 1.5)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    network = NetworksManager(device, state_dim, action_dim, HIDDEN_DIM, None)    
    network.load_weights(WEIGHTS_FOLDER)
    network.eval()
    
    episode_rewards = [] 
    for episode in tqdm(range(TEST_EPISODES)):
        state = env.reset()
        if ENV == '2d': state = state[0]
        episode_reward = 0
        step = 0
        while step < MAX_STEPS:
            action = network.policy_net.get_action(state).detach()
            next_state, reward, done, *_ = env.step(action.numpy())
        
            state = next_state
            episode_reward += reward
            step += 1
            
            if done: break
        episode_rewards.append(episode_reward)
        print('episode', episode, 'reward: ', episode_reward)
    
    print('Avg reward over', TEST_EPISODES, '->', np.mean(episode_rewards))


if __name__ == '__main__':
    main()