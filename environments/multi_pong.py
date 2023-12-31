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

        self.agents = ["paddle_1", "paddle_2"]

        # Game settings
        self.width = width
        self.height = height
        self.paddle_height = paddle_height

        self.paddles = {"paddle_1": self.height // 2 + 2, "paddle_2": self.height // 2 - 2}

        self.balls = {"ball_1": {"position": [self.width // 2 - 1, self.height // 2], "direction":[random.uniform(0.5, 1), random.uniform(0.5, 1)]},
                      "ball_2": {"position": [self.width // 2 + 1, self.height // 2], "direction":[random.uniform(0.5, 1), random.uniform(0.5, 1)]}}

        # Observation space: position of paddle and ball
        self.observation_space = spaces.Box(low=-1 * max(self.width, self.height), high=max(self.width, self.height), shape=(10,), dtype=np.float16)

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
    
    def __move_ball(self, ball_pos, ball_direction):
        ball_pos[0] += ball_direction[0]
        ball_pos[1] += ball_direction[1]
        if ball_pos[1] <= 0 or ball_pos[1] >= self.height - 1:
            ball_direction[1] *= -1
        if ball_pos[0] <= 0:
            ball_direction[0] *= -1
        rewards = {paddle: 0 for paddle in self.paddles.keys()}
        for paddle in self.paddles.keys():
            if ball_pos[0] >= self.width - 1 and self.paddles[paddle] <= ball_pos[1] < self.paddles[paddle] + self.paddle_height:
                ball_direction[0] *= -1
                rewards[paddle] = 1
                ball_pos[0] += ball_direction[0]
                ball_pos[1] += ball_direction[1]
        return ball_pos, ball_direction, rewards
    
    def check_done(self, ball_pos):
        if ball_pos[0] >= self.width:
            return True
        return False
    
    def __get_observation(self, paddle):

        rel_paddle_pos = self.get_relative_position((self.width - 1, self.paddles[paddle]))

        obs = np.array([rel_paddle_pos[0], rel_paddle_pos[1]], dtype=np.float16)

        for ball in self.balls.keys():
            rel_ball_pos = self.get_relative_position(self.balls[ball]["position"])
            ball_direction = self.balls[ball]["direction"]
            ball_obs = np.array([rel_ball_pos[0], rel_ball_pos[1], ball_direction[0], ball_direction[1]])
            obs = np.append(obs, ball_obs)
        return obs

    def step(self, actions):
        # Update paddle position based on action

        for paddle in actions.keys():
            action = actions[paddle]
            if action == 1 and self.paddles[paddle] > 0:
                self.paddles[paddle] -= 1
            elif action == 2 and self.paddles[paddle] < self.height - self.paddle_height:
                self.paddles[paddle] += 1

        rewards = {paddle: 0 for paddle in self.paddles.keys()}
        for ball in self.balls.keys():
            ball_pos, ball_direction, new_rewards = self.__move_ball(self.balls[ball]["position"], self.balls[ball]["direction"])
            self.balls[ball] = {"position":ball_pos, "direction":ball_direction}
            for paddle in new_rewards.keys():
                rewards[paddle] += new_rewards[paddle]

        done = False
        if any([self.check_done(self.balls[ball]["position"]) for ball in self.balls.keys()]):
            done = True
            rewards = {paddle: -1 for paddle in self.paddles.keys()}  # Negative reward for losing the ball

        dones = {paddle: done for paddle in self.paddles.keys()}

        obs = {paddle: self.__get_observation(paddle) for paddle in self.paddles.keys()}

        return obs, rewards, dones, {}

    def reset(self):
        # Reset the game state
        self.paddles = {"paddle_1": self.height // 2 + 2, "paddle_2": self.height // 2 - 2}

        self.balls = {"ball_1": {"position": [self.width // 2 - 1, self.height // 2], "direction":[random.uniform(0.5, 1), random.uniform(0.5, 1)]},
                      "ball_2": {"position": [self.width // 2 + 1, self.height // 2], "direction":[random.uniform(0.5, 1), random.uniform(0.5, 1)]}}

        obs = {paddle: self.__get_observation(paddle) for paddle in self.paddles}
        return obs

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        # Draw game board in console
        board = [[' ' for _ in range(self.width + 1)] for _ in range(self.height + 1)]
        paddle_symbolds = ['|', '/', '*', '+']
        for i, paddle in enumerate(self.paddles.keys()):
            for j in range(self.paddle_height):
                paddle_pos = self.paddles[paddle]
                board[paddle_pos + j][self.width - 1] = paddle_symbolds[i]

        
        for ball in self.balls.keys():
            ball_pos = self.balls[ball]["position"]
            board[int(ball_pos[1])][int(ball_pos[0])] = 'O'

        # Print game board
        print("-" * (self.width + 2))
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
        observations, rewards, dones, info = self.env.step(action)
        if self.step_count >= self.max_steps:
            done = True
            info['time_limit_reached'] = True
        return observations, rewards, dones, info