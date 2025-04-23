# LukaBot

### Overview
LukaBot is a reinforcement learning (RL) project that simulates a basketball shooting scenario inspired by NBA player Luka Dončić. The goal is to train an RL agent to shoot a basketball from the 3-point line (7.24 meters from the hoop) and successfully make shots into a hoop at a height of 3.05 meters (10 ft - standard NBA hoop height). The environment is built using the Gym interface, and the agent is trained using the Soft Actor-Critic (SAC) algorithm from Stable Baselines3.

### Setup
The agent learns to optimize two continuous actions:
- Velocity (m/s): The initial speed of the shot, constrained between 5 and 30 m/s.
- Angle (degrees): The shooting angle, constrained between 20 and 80 degrees.

The observation (state) space consists of: 
- Distance to hoop (m) limited from [0, 10]
- Final x position (m) limited from [0, 10]
- Final y position (m) limited from [0, 5]
- Shots taken limited from [0, 10]
- Successful shots limited from [0, 10]

Each episode consists of 10 shot attempts, and the agent receives rewards based on the shot's outcome (e.g., making the shot, getting close to the hoop, or missing). The reward function used assigns a significant missed penalty for shots that are way off target, and assigns a progressively positive reward for shots near the goal.The project includes visualization of shot trajectories using Matplotlib and detailed logging for debugging and analysis.

### Limitations & Future Improvement
- Currently, the project is strictly 2-D, with no account of the z-axis. In basketball terms, this is similar to only shooting from one spot at the top of the key. In future renditions, I want the agent to be able to shoot from anywhere on the court.
- Optimizing my reward function and SAC hyperparameters for fastest and most accurate convergence
- In the original trial of this project, I was using Stable Baselines3 PPO. However, this model failed to learn, despite a lot of hyperparamter tweaking and revisions of the LukaBot class implementation. I want to find out exactly *why* this happens with PPO but not SAC.
- For now, I'm just using matplotlib to plot shot trajectory. A more elaborate visualization or simulation would be ideal.

### Resources & Help
- https://gymnasium.farama.org/api/env/#gymnasium.Env.step
- https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
- Generative AI: ChatGPT, Claude, Gemini, Grok
- Notes from COGS 188, taught by Professor Jason Flesicher
- Assistance from Professor Kyle Shannon

