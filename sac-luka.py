# importing necessary libraries
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
import logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("luka_bot.log")
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# defining constants (in terms of meters)
GRAVITY = 9.81
HOOP_X = 7.24 
HOOP_Y = 3.05  
BASKET_RADIUS = 0.2286 
BALL_RADIUS = 0.12065 


def feet_to_meters(feet):
    '''
    Helper function to convert feet to meters
    '''
    return feet * 0.3048

# Class defining the LukaBot environment
class LukaBotEnv(gym.Env):
    '''
    Custom Environment that follows gym interface for LukaBot shooting simulation.
    Observation space: [Distance to hoop (m), final x position (m), final y position (m), shots taken, successful shots]
    Observation space limited to [0, 10] m for distance and x position, [0, 5] m for y position, [0, 10] for shots taken and successful shots.
    Action space: [Velocity (m/s) and angle (degrees)]
    Action space limited to [5, 30] m/s for velocity and [20, 80] degrees for angle.
    Each episode consists of 10 shot attempts.
    '''
    def __init__(self, player_height_ft=6.5): # Luka is 6'6
        super(LukaBotEnv, self).__init__()

        # Training trackers
        self.current_episode = 0
        self.total_rewards = 0
        self.successful_shots = 0
        self.shots_taken = 0 
        self.max_shots_per_episode = 10

        # tracking shot and state information
        self._last_shot_info = {}
        self.state_history = []

        # Environment attributes
        self.release_point_height = feet_to_meters(player_height_ft + 1) 
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0]),  high=np.array([10, 10, 5, self.max_shots_per_episode, self.max_shots_per_episode]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([5, 20]), high=np.array([30, 80]), dtype=np.float32)
        self.state = None
        self.reset()


    def reset(self, seed=None, options=None):
        '''
        Reset the environment to an initial state.
        This method is called at the beginning of each episode.
        Set the state to the initial values, reset the shot counters, and log the episode number.
        '''
        self.current_episode += 1
        self.total_rewards = 0
        self.successful_shots = 0
        self.shots_taken = 0  

        if self.current_episode % 1000 == 0:
            logger.info(f"Episode {self.current_episode}")
            logger.info(f"Total Rewards: {self.total_rewards}")
            logger.info(f"Successful Shots: {self.successful_shots}")

        self._last_shot_info = {
            'velocity': None,
            'angle': None,
            'distance': None,
            'height': None,
            'reward': None
        }
        self.state = np.array([HOOP_X, 0, self.release_point_height, self.shots_taken, self.successful_shots], dtype=np.float32)
        return self.state, {}



    def calculate_reward(self, x, y):
        '''
        Helper function to calculate the reward based on the shot's outcome.
        '''
        if y <= 0:
            return -2 + 0.5 * x # Dynamic penalty for missing the hoop

        x_distance_to_hoop = abs(x - HOOP_X)
        distance_to_hoop = np.sqrt((x - HOOP_X)**2 + (y - HOOP_Y)**2)
        if x_distance_to_hoop <= (BASKET_RADIUS - BALL_RADIUS) and abs(y - HOOP_Y) <= 0.05:
            if self.successful_shots > 0 and self._last_shot_info.get('reward', 0) >= 10:
                bonus = 2 # bonus for consecutive makes
            else:
                bonus = 0 
            return 10 + bonus
        
        # Partial Reward if the shot is close but doesn't go in
        if distance_to_hoop < 0.1:  
            return 7
        elif distance_to_hoop < 0.2:  
            return 5
        elif distance_to_hoop < 0.3:  
            return 3
        elif distance_to_hoop < 0.5:  
            return 2 
        elif distance_to_hoop < 1.0:  
            return 1  

        # Negative reward for ball far away from the hoop
        return -1 + 0.5 * x


    def step(self, action):
        '''
        Simulate a shot based on the action taken (velocity and angle).
        The shot is simulated for a maximum of 3 seconds or until the ball hits the ground or goes through the hoop.
        The method returns the new state, reward, done flag, and additional info about the last shot.
        The state is updated to reflect the final distance to the hoop, final x and y positions, shots taken, and successful shots.
        '''
        velocity, angle = action
        logger.info(f"Action received: {action}")
        angle = np.deg2rad(angle)  
        v_x = velocity * np.cos(angle)  # Horizontal velocity
        v_y = velocity * np.sin(angle)  # Vertical velocity
        t = 0  
        dt = 0.01 
        max_time = 3  
        x = 0
        y = self.release_point_height

        # Simulate the shot, stop if ball hits the ground or scores
        while t < max_time:
            x = v_x * t  
            y = self.release_point_height + v_y * t - 0.5 * GRAVITY * t**2  
            if y <= 0: 
                break
            elif abs(x - HOOP_X) <= (BASKET_RADIUS - BALL_RADIUS) and abs(y - HOOP_Y) <= 0.03: 
                break
            t += dt

        reward = self.calculate_reward(x, y)
        self.total_rewards += reward

        # Update successful shots and shots taken
        if reward >= 10:  
            self.successful_shots += 1
        self.shots_taken += 1

        # Check if episode is done (10 shot attempts)
        done = (self.shots_taken >= self.max_shots_per_episode)

        # Update last shot info
        self._last_shot_info.update({
            'velocity': velocity,
            'angle': np.rad2deg(angle),
            'distance': x,
            'height': y,
            'reward': reward
        })

        logger.info(f"Shot {self.shots_taken}/{self.max_shots_per_episode} Details: "
                    f"Velocity={velocity:.2f}m/s, "
                    f"Angle={np.rad2deg(angle):.2f}°, "
                    f"Distance={x:.2f}m, "
                    f"Height={y:.2f}m, "
                    f"Reward={reward}")

        # After the shot, reset the ball's position to the starting point for the next shot
        if not done:
            distance_to_hoop = HOOP_X  
            x = 0  
            y = self.release_point_height  
        else:
            distance_to_hoop = np.sqrt((x - HOOP_X)**2 + (y - HOOP_Y)**2) 

        self.state = np.array([distance_to_hoop, x, y, self.shots_taken, self.successful_shots], dtype=np.float32)
        self.state_history.append(self.state)
        logger.info(f"Updated state: {self.state}, Reward: {reward}, Done: {done}")
        return self.state, reward, done, False, self._last_shot_info


def train_and_visualize(total_timesteps=1000000):
    """
    Train the SAC agent and visualize shot trajectories.
    """
    # Create environment
    env = LukaBotEnv()
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    # Train SAC model
    model = SAC("MlpPolicy", env, learning_rate=0.0003, buffer_size=100000, batch_size=256, tau=0.005, gamma=0.99, ent_coef="auto", verbose=1)
    model.learn(total_timesteps=total_timesteps, log_interval=100)

    # Test the trained model
    test_agent(env, model, episodes=5)


def test_agent(env, model, episodes=5):
    """
    Test the trained SAC agent and visualize shot trajectories.
    """
    for episode in range(episodes):
        obs = env.reset() 
        done = False
        episode_rewards = 0
        episode_successful_shots = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)  # Use deterministic actions for testing
            print(f"Predicted Action: {action}")
            
            next_obs, reward, done, info = env.step(action)
            reward = reward[0]  # Shape (1,) -> scalar
            done = done[0]      # Shape (1,) -> scalar
            info = info[0]      # Shape (1,) -> dict
            episode_rewards += reward

            if reward >= 10:  # scores
                episode_successful_shots += 1
            print(f"Observation After Action: {next_obs}, Reward: {reward}")
            print(f"\nShot {int(next_obs[0][3])} in Episode {episode + 1}:")
            print(f"Shot Velocity: {info['velocity']:.2f} m/s")
            print(f"Shot Angle: {info['angle']:.2f} degrees")
            print(f"Distance: {info['distance']:.2f} m")
            print(f"Height: {info['height']:.2f} m")
            print(f"Reward: {reward}")
            
            # Visualize the shot
            plot_shot(info['velocity'], info['angle'])
            obs = next_obs
        
        # Log episode summary
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"Total Rewards: {episode_rewards}")
        print(f"Successful Shots: {episode_successful_shots}/{env.get_unwrapped()[0].max_shots_per_episode}")


def plot_shot(velocity, angle):
    """
    Create a 2D visualization of a basketball shot trajectory.
    """
    # Simulation parameters
    release_height = feet_to_meters(7.5)
    angle_rad = np.radians(angle)
    t_values = np.linspace(0, 2, num=100)
    
    # Calculate trajectory
    x_values = velocity * np.cos(angle_rad) * t_values
    y_values = (release_height + velocity * np.sin(angle_rad) * t_values - 0.5 * GRAVITY * t_values**2)
    
    # Plot trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label=f"v={velocity:.2f}, θ={angle:.2f}°")
    plt.scatter([HOOP_X], [HOOP_Y], color='red', label='Hoop')
    plt.xlim(0, 10)
    plt.ylim(0, 5)
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.title("Basketball Shot Trajectory")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the training
if __name__ == "__main__":
    train_and_visualize()