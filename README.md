# Pin-Point Autonomous Lunar Landing
# Autonomous Precision Landing with ISAC and AISAC

## Introduction
This repository presents an implementation of the Improved Soft Actor-Critic (ISAC) architecture for performing autonomous precision landing—pin-point landing—on a planetary environment, specifically focusing on the lunar 3D environment,we also provide an experimentation of ISAC in the 2D enviroment lunar landing. Additionally, we introduce modifications to ISAC, resulting in Asyncronous-ISAC (our contribution), aimed at enhancing sample efficiency and performance in trajectory recalculation for fault recovery scenarios.
Our contribution to the field, AISAC, extends ISAC's capabilities by introducing asynchronous processing. This approach involves multiple sub-agents, each responsible for rollout without updating the network, thereby leveraging parallel processing for accelerated training.

## Abstracts of Implemented Papers
- **Deep Reinforcement Learning for Pin-Point Autonomous Lunar Landing: Trajectory Recalculation for Obstacle Avoidance:** This work aims to present a method to perform autonomous precision landing—pin-point landing—on a planetary environment and perform trajectory recalculation for fault recovery where necessary. In order to achieve this, we choose to implement a Deep Reinforcement Learning—DRL—algorithm, i.e. the Soft Actor-Critic—SAC—architecture. In particular, we select the lunar environment for our experiments, which we perform in a simulated environment, exploiting a real-physics simulator modeled by means of the Bullet/PyBullet physical engine. We show that the SAC algorithm can learn an effective policy for precision landing and trajectory recalculation if fault recovery is made necessary—e.g. for obstacle avoidance.
- **Improved Soft Actor-Critic: Mixing Prioritized Off-Policy Samples with On-Policy Experience:** Building upon SAC, ISAC introduces modifications to enhance sample efficiency and stability. These include a new prioritization scheme for selecting better samples from the experience replay buffer and a mixture of prioritized off-policy data with the latest on-policy data for training.

## Implementation Details
- **Environment:** Our experiments are conducted within a simulated lunar environment utilizing the robust Bullet/PyBullet physics engine. The 3D environment was generously provided by the original authors.
  - `python3 ISAC_train_torch.py`: Training script for ISAC.
  - `python3 AISAC_train_torch.py`: Training script for AISAC.
  - `python3 SAC_train_torch.py`: Training script for standard SAC.

## Getting Started
To initiate training for the respective algorithms, execute the following commands:
```bash
python3 ISAC_train_torch.py
python3 AISAC_train_torch.py
python3 SAC_train_torch.py
```

In our training framework, to understand the results of our esperimentation, it's crucial to understand the distinction between local episodes and global episodes:

- **Local Episode:** This refers to the episode counter specific to the main agent. It tracks the number of episodes the main agent has completed independently.
  
- **Global Episode:** On the other hand, global episodes represent a shared counter utilized across all agents, particularly in multi-processing setups. While the main agent may have completed 100 local episodes, the global episode count might exceed 100 due to the involvement of other agents. This shared counter ensures synchronization and coordination among all agents during training.

in our graphs, Local episode is described simply as Episode.
## Results
Results are presented in comparison with the standard SAC algorithm. Graphs and descriptions are provided below:
![Lunar Landing](images/lunar_landing.png)
