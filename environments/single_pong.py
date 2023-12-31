import gym
from gym import spaces
import numpy as np
import random

class PongEnv(gym.Env):
    """
    Custom Environment for Pong game compatible with OpenAI Gym.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, width=20, height=10, paddle_height=3):
        super(PongEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 = Stay, 1 = Up, 2 = Down
        self.action_space = spaces.Discrete(3)

        # Game settings
        self.width = width
        self.height = height
        self.paddle_height = paddle_height
        self.paddle_pos = self.height // 2
        self.ball_pos = [self.width // 2, self.height // 2]
        self.ball_direction = [random.uniform(0.5, 1), random.uniform(0.5, 1)]

        # Observation space: position of paddle and ball
        self.observation_space = spaces.Box(low=-1 * max(self.width, self.height), high=max(self.width, self.height), shape=(6,), dtype=np.float16)

    def get_relative_position(self, object_pos, invert_x=False):
        """
        Calculate the relative position of an object to the center of the field.

        Args:
            object_pos (tuple): A tuple (x, y) representing the position of the object.
            invert_x (bool): If True, invert the x-axis value.

        Returns:
            tuple: A tuple (x_relative, y_relative) representing the relative
                position of the object to the center of the field. Positive values
                mean the object is to the right or above the center, negative
                values mean to the left or below. If invert_x is True, the x-axis
                values are inverted.
        """
        center_x = self.width / 2
        center_y = self.height / 2

        x_relative = object_pos[0] - center_x
        y_relative = object_pos[1] - center_y

        if invert_x:
            x_relative = -x_relative

        return (x_relative, y_relative)


    def step(self, action):
        # Update paddle position based on action
        if action == 1 and self.paddle_pos > 0:
            self.paddle_pos -= 1
        elif action == 2 and self.paddle_pos < self.height - self.paddle_height:
            self.paddle_pos += 1

        reward = 0

        # Update ball position and check for collision
        self.ball_pos[0] += self.ball_direction[0]
        self.ball_pos[1] += self.ball_direction[1]
        if self.ball_pos[1] <= 0 or self.ball_pos[1] >= self.height - 1:
            self.ball_direction[1] *= -1
        if self.ball_pos[0] <= 0:
            self.ball_direction[0] *= -1
        if self.ball_pos[0] >= self.width - 1 and self.paddle_pos <= self.ball_pos[1] < self.paddle_pos + self.paddle_height:
            self.ball_direction[0] *= -1
            self.ball_pos[0] += self.ball_direction[0]
            reward = 1

        done = False
        if self.ball_pos[0] >= self.width:
            done = True
            reward = -1  # Negative reward for losing the ball

        rel_ball_pos = self.get_relative_position(self.ball_pos)
        rel_paddle_pos = self.get_relative_position((self.width - 1, self.paddle_pos))

        obs = np.array([rel_paddle_pos[0], rel_paddle_pos[1], rel_ball_pos[0], rel_ball_pos[1], self.ball_direction[0], self.ball_direction[1]], dtype=np.float16)

        return obs, reward, done, {}

    def reset(self):
        # Reset the game state
        self.paddle_pos = self.height // 2
        self.ball_pos = [self.width // 2, self.height // 2]
        self.ball_direction = [random.uniform(0.5, 1), random.uniform(0.5, 1)]

        rel_ball_pos = self.get_relative_position(self.ball_pos)
        rel_paddle_pos = self.get_relative_position((self.width - 1, self.paddle_pos))

        obs = np.array([rel_paddle_pos[0], rel_paddle_pos[1], rel_ball_pos[0], rel_ball_pos[1], self.ball_direction[0], self.ball_direction[1]], dtype=np.float16)
        return obs

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        # Draw game board in console
        board = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        for i in range(self.paddle_height):
            board[self.paddle_pos + i][self.width - 1] = '|'
        board[int(self.ball_pos[1])][int(self.ball_pos[0])] = 'O'

        # Print game board
        print('\n'.join([''.join(row) for row in board]))
        print("-" * (self.width + 2))
        print()

    def close(self):
        pass

class MaxEpisodeStepsWrapper(gym.Wrapper):
    def __init__(self, env, max_steps):
        super().__init__(env)
        self.max_steps = max_steps
        self.step_count = 0

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.step_count += 1
        observation, reward, done, info = self.env.step(action)
        if self.step_count >= self.max_steps:
            done = True
            info['time_limit_reached'] = True
        return observation, reward, done, info