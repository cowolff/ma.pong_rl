import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from single_pong_multiple_balls import PongEnv, MaxEpisodeStepsWrapper
import time

# Assuming PongEnv is already defined as shown in previous examples

# Create the environment
env = make_vec_env(lambda: MaxEpisodeStepsWrapper(PongEnv(), 2048), n_envs=4)

# Initialize the agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
total_timesteps = 2000000  # Adjust this to train for more or less time
model.learn(total_timesteps=total_timesteps)

# Save the trained model
model.save("ppo_pong")

# Close the environment
env.close()

def render_model_results(model_path, env, num_episodes=5):
    """
    Renders the results of a trained model in the given environment.

    :param model_path: Path to the saved model.
    :param env: The Gym environment to use.
    :param num_episodes: Number of episodes to render.
    """
    # Load the trained model
    model = PPO.load(model_path)

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            env.render()
            time.sleep(0.1)
            if done:
                break

# Usage example
env = PongEnv()  # Replace with your environment
model_path = "ppo_pong"  # Replace with your model's file path
render_model_results(model_path, env)