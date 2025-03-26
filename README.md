# LukaBot

Overview
LukaBot is a reinforcement learning (RL) project that simulates a basketball shooting scenario inspired by NBA player Luka Dončić. The goal is to train an RL agent to shoot a basketball from the 3-point line (7.24 meters from the hoop) and successfully make shots into a hoop at a height of 3.05 meters. The environment is built using the Gym interface, and the agent is trained using the Soft Actor-Critic (SAC) algorithm from Stable Baselines3.

The agent learns to optimize two continuous actions:

Velocity (m/s): The initial speed of the shot, constrained between 5 and 30 m/s.

Angle (degrees): The shooting angle, constrained between 20 and 80 degrees.

Each episode consists of 10 shot attempts, and the agent receives rewards based on the shot's outcome (e.g., making the shot, getting close to the hoop, or missing). The project includes visualization of shot trajectories using Matplotlib and detailed logging for debugging and analysis.
