#!/usr/bin/env python3
import torch
import os
import concurrent.futures
import psutil
from rwlock import RWLock
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import deque
import setproctitle
import sys 
import os
import wandb

# Hyperparameters:
MAX_EPISODES = 1000
MAX_STEPS = 100
REPLAY_BUFFER_SIZE = 10_000
REPLAY_BUFFER_THRESHOLD = 0.5
BATCH_SIZE = 64
HIDDEN_DIM = 256
N_ASYNC_PROCESSES = 2
ACTION_REPEAT = 50 # Number of times to repeat each action in the 3d environment

ENV = '3d' # '2d' or '3d
WEIGHTS_FOLDER = 'AISAC_weights_'+ENV
LOAD_WEIGHTS = False
RENDER = False 
WANDB_LOG = True
WANDB_RUN_NAME = 'lander-'+ENV+'-aisac'
USE_GPU_IF_AVAILABLE = True 

# TODO organize better the repo to avoid this:
setproctitle.setproctitle("AutonomousLandingProcess")
sys.path.append('.')
sys.path.append('..')

from prioritized_replay_buffer import ReplayBuffer
from async_agent import AsyncAgent
from model import NetworksManager
print('OK! All imports successful!')

def get_resource_usage():
    # Get system resource usage
    cpu_percent = psutil.cpu_percent(interval=1)
    virtual_memory = psutil.virtual_memory()
    swap_memory = psutil.swap_memory()
    disk_usage = psutil.disk_usage('/')

    # Create a string with the resource information
    resource_info = f"CPU Usage: {cpu_percent}%\n" \
                    f"RAM Usage: {virtual_memory.percent}%\n" \
                    f"Swap Usage: {swap_memory.percent}%\n" \
                    f"Disk Usage: {disk_usage.percent}%\n"

    return resource_info

def save_to_file(file_path, content):
    # Save the content to a text file
    with open(file_path, 'w') as file:
        file.write(content)


def main():
    print("The Main Process PID is: ", os.getpid())
    torch.manual_seed(42)
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if USE_GPU_IF_AVAILABLE else torch.device("cpu")
    print('Device set to : ' + str(torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu' ))
    async_device = torch.device("cpu")
    
    if WANDB_LOG:
        wandb.login(key='efa11006b3b5487ccfc221897831ea5ef2ff518f')
        wandb.init(project='lunar_lander', 
                    name=WANDB_RUN_NAME,
                    config={
                        'env': ENV,
                        'max_episodes': MAX_EPISODES,
                        'max_steps': MAX_STEPS,
                        'replay_buffer_size': REPLAY_BUFFER_SIZE,
                        'replay_buffer_threshold': REPLAY_BUFFER_THRESHOLD,
                        'batch_size': BATCH_SIZE,
                        'hidden_dim': HIDDEN_DIM,
                        'load_weights': LOAD_WEIGHTS,
                        'device': device.type,
                        'action_repeat': ACTION_REPEAT,
                        'n_async_processes': N_ASYNC_PROCESSES
                        }
                    )

    if ENV == '3d':
        # from lander_gym_env import LanderGymEnv # Load simplified environment - no atmospheric disturbances
        from lander_gym_env_with_gusts import LanderGymEnv # Load environment with gusts
        env = LanderGymEnv(renders=RENDER, actionRepeat=ACTION_REPEAT)
    else:  
        render_mode = 'human' if RENDER else None
        env = gym.make("LunarLander-v2", render_mode=render_mode, continuous = True, gravity = -10.0, enable_wind = False, wind_power = 15.0, turbulence_power = 1.5)

    print('OK! Environment configuration successful!')
    state_dim = env.observation_space.shape[0]
    print("Size of state space -> {}".format(state_dim))
    action_dim = env.action_space.shape[0]
    print("Size of action space -> {}".format(action_dim))
    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]
    print("max value of action -> {}".format(upper_bound))
    print("min value of action -> {}".format(lower_bound))
    
    frame_idx = 0
    episode_rewards = deque(maxlen=100)

    rwlock = RWLock()
    #weights_file_lock = InterProcessLock("weights/AISAC_weights/")#manager_shared_data.Lock() # Shared lock for file access
    network = NetworksManager(device, state_dim, action_dim, HIDDEN_DIM, rwlock)
    
    if not os.path.exists(WEIGHTS_FOLDER):
        os.makedirs(WEIGHTS_FOLDER)
    if LOAD_WEIGHTS: network.load(WEIGHTS_FOLDER)
    network._save_sync(WEIGHTS_FOLDER)
    
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, threshold=REPLAY_BUFFER_THRESHOLD)
    local_buffer = [] # cumulate the transitions here and at the end of each episode push the cumulative reward (rho) to replay_buffer
    last_infusion_episode = 0
    #TODO: variaable that says if the network is updated. if yes, agents will update theirs network, otherwise its useless
        
    # Define shared stuff
    global_episode_counter = torch.multiprocessing.Value('i', 0)
    delayed_buffer = torch.multiprocessing.Queue(maxsize= REPLAY_BUFFER_SIZE)
    delayed_buffer_available = torch.multiprocessing.Event()
    delayed_buffer_available.set()

    # Specify the file path where you want to save the information
    file_path = "resource_usage.txt"


    resource_info = get_resource_usage()
    # Save the information to the specified file
    save_to_file(file_path, resource_info)

    
    agents = [AsyncAgent(i, async_device, global_episode_counter, delayed_buffer, delayed_buffer_available, HIDDEN_DIM, WEIGHTS_FOLDER, MAX_EPISODES, ENV, state_dim, action_dim, rwlock) for i in range(N_ASYNC_PROCESSES)]
    [agent.start() for agent in agents]
    print("All the ",N_ASYNC_PROCESSES," Agents are ready!")
  
    # Training loop:
    episode = None
    while global_episode_counter.value < MAX_EPISODES:
        state = env.reset()
        if ENV == '2d': state = state[0]
        episode_reward = 0
        with global_episode_counter.get_lock():
            global_episode_counter.value += 1
            episode = global_episode_counter.value
        print(f'''Agent MAIN\tEpisode {episode} starting at frame_idx {frame_idx}''')
        step = 0
        while step <= MAX_STEPS:
            if global_episode_counter.value > 100:
                action = network.policy_net.get_action(state).detach()
                next_state, reward, done, *_ = env.step(action.numpy())
            else: 
                action = env.action_space.sample()
                next_state, reward, done, *_ = env.step(action)

            local_buffer.append( [state, action, reward, next_state, done] )
            replay_buffer.set_latest_transition(local_buffer[-1])

            state = next_state
            episode_reward += reward
            frame_idx += 1
            step += 1
            if len(replay_buffer) >= BATCH_SIZE:
                network.update(replay_buffer, BATCH_SIZE)
                #await loop.run_in_executor(pool, network.save_async,weights_filename)
            
            if done:
                break
            
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards)
        if WANDB_LOG: wandb.log({"episode": episode, "frame": frame_idx, "episode_reward": episode_reward, "avg_reward": avg_reward})
        print(f'''Agent MAIN:\tEpisode {episode} FINISHED after {step} steps ==> Episode Reward: {episode_reward:3f} / Avg Reward: {avg_reward:.3f}''')
        
        for transition in local_buffer: transition.append(episode_reward) # push the cumulative reward to the replay buffer
        # maybe the push here, better to do it not every episode so that delay_local_buffer is not always taken by some agent
        # having an overall improvement in performance since we do less updates (and so less get_lock() and less waiting)
        for transition in local_buffer: 
            if delayed_buffer.full(): delayed_buffer.get() # removing the oldest transition from the queue
            delayed_buffer.put(tuple(transition))
        local_buffer = []
        
        # The idea is to delay the infusion  of new experience to the replay_buffer 
        # avoiding overfitting so that the network will be trained more using old experience
        if global_episode_counter.value - last_infusion_episode > 2:
            print("Main Agent is updating the shared replay buffer...")
            
            last_infusion_episode = global_episode_counter.value
            delayed_buffer_available.clear()
            while not delayed_buffer.empty():
                transition = delayed_buffer.get()
                replay_buffer.push_transition(*transition)
            delayed_buffer_available.set()
                
            print(f'''Replay buffer updated ==> new len: {len(replay_buffer)}''')

        with concurrent.futures.ThreadPoolExecutor() as pool: # Run in a custom thread pool
            # Get resource usage information
            resource_info = get_resource_usage()
            pool.submit(network.save_async, WEIGHTS_FOLDER)
            pool.submit(save_to_file, file_path, resource_info)
            #await loop.run_in_executor(pool,network.save_async,weights_filename)
    
    [agent.terminate() for agent in agents] # delete all the agents when Main Agent finished
    replay_buffer.clear_buffer()
    delayed_buffer.cancel_join_thread()
    del network # Delete the model to free up memory
    torch.cuda.empty_cache() # Release GPU memory not in use by PyTorch

    
if __name__ == '__main__':
    main()





