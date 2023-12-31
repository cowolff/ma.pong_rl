# Import required library
import random
import numpy as np
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv
import functools
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

class Paddle():
	def __init__(self, x, y, speed=1, size=(6, 2), borders=(-280, 280), side="left") -> None:
		self.position = [x, y]
		self.speed = speed
		self.size = size
		self.borders = borders
		self.side = side
		if side not in ["left", "right"]:
			raise ValueError(f"Side must be either 'left' or 'right', not {side}")

	def __checkBorder(self):
		if self.position[1] > self.borders[1]:
			self.position[1] = self.borders[1]
		elif self.position[1] < self.borders[0]:
			self.position[1] = self.borders[0]
	
	def move(self, action):
		if np.where(action == 1)[0] == 0:
			self.position[1] += self.speed
		elif np.where(action == 1)[0] == 1:
			self.position[1] -= self.speed
		else:
			pass
		self.__checkBorder()

	def getPos(self):
		return self.position

class Ball():
	def __init__(self, speed, borders=(-280, 280)) -> None:
		self.position = [0, 0]
		self.speed = speed
		self.direction = [random.uniform(-1, 1), random.uniform(-1, 1)]
		self.borders = borders

	def __checkBorder(self):
		if self.position[1] > self.borders[1]:
			self.position[1] = self.borders[1]
			self.direction[1] *= -1
		elif self.position[1] < self.borders[0]:
			self.position[1] = self.borders[0]
			self.direction[1] *= -1
	
	def move(self):
		self.position[0] += self.speed * self.direction[0]
		self.position[1] += self.speed * self.direction[1]
		self.__checkBorder()

	def getPos(self):
		return self.position


class PongGameEnv(ParallelEnv):

	metadata = {
        "name": "pong_v0",
    }

	def __init__(self, speed=1, paddle_size=(6, 2), borders=(-280, 280), length=200) -> None:
		self.possible_agents = ["left", "right"]
		self.speed = speed
		self.paddle_size = paddle_size
		self.length = length
		self.borders = borders
		self.left_paddle = Paddle(-self.length + 10, 0, speed, paddle_size, borders)
		self.right_paddle = Paddle(self.length - 10, 0, speed, paddle_size, borders)
		self.ball = Ball(speed, borders)
		self.score = [0, 0]
		self.timestep = 0
		self.done = False
		self.action_spaces = {agent: Discrete(3) for agent in self.possible_agents}
		self.observation_spaces = {agent: Box(low=-50, high=50, shape=(6,)) for agent in self.possible_agents}

	def reset(self, seed=None, options=None):
		self.agents = copy(self.possible_agents)
		self.left_paddle = Paddle(-self.length + 10, 0, self.speed, self.paddle_size, self.borders)
		self.right_paddle = Paddle(self.length - 10, 0, self.speed, self.paddle_size, self.borders)
		self.ball = Ball(self.speed, self.borders)
		self.score = [0, 0]
		self.done = False
		observation = {agent: self.get_observation(agent) for agent in self.agents}
		infos = {agent: {} for agent in self.agents}
		return observation, infos
	
	def collision(self, paddle_pos, paddle_size, ball_pos, ball_vector):
		x1, y1 = paddle_pos[0], paddle_pos[1]
		x2, y2 = ball_pos[0], ball_pos[1]
		x_size1, y_size1 = paddle_size[0], paddle_size[1]
		x_move, y_move = ball_vector[0], ball_vector[1]

		# Calculate the future position of the second object
		future_x2 = x2 + x_move
		future_y2 = y2 + y_move

		# Check if the two objects will collide
		if (
			x1 < future_x2 + x_size1 and
			x1 + x_size1 > future_x2 and
			y1 < future_y2 + y_size1 and
			y1 + y_size1 > future_y2
		):
			return True
		else:
			return False
		
	def step(self, actions):
		self.left_paddle.move(actions["left"])
		self.right_paddle.move(actions["right"])
		self.ball.move()
		self.timestep += 1

		reward = {agent: 0 for agent in self.possible_agents}
		for agent in self.possible_agents:
			if self.collision(self.left_paddle.getPos(), self.left_paddle.size, self.ball.getPos(), self.ball.direction):
				reward["left"] = 1
				self.ball.direction[0] *= -1
				self.ball.move()
			elif self.collision(self.right_paddle.getPos(), self.right_paddle.size, self.ball.getPos(), self.ball.direction):
				reward["right"] = 1
				self.ball.direction[0] *= -1
				self.ball.move()

		terminations = {agent: False for agent in self.possible_agents}
		truncations = {agent: False for agent in self.possible_agents}
		if self.ball.getPos()[0] > self.length:
			terminations["right"] = True
			reward["right"] = -1
		elif self.ball.getPos()[0] < -self.length:
			terminations["left"] = True
			reward["left"] = -1
		observation = {agent: self.get_observation(agent) for agent in self.agents}
		infos = {agent: {} for agent in self.agents}
		return observation, reward, terminations, truncations, infos

	def render(self):
		grid = np.full((self.length * 2 + 2, abs(self.borders[0]) + abs(self.borders[1]) + 2), 0)
		grid[self.left_paddle.getPos()[0] + self.length, self.left_paddle.getPos()[1] + abs(min(self.borders))] = 50
		grid[self.right_paddle.getPos()[0] + self.length, self.right_paddle.getPos()[1]+ abs(min(self.borders))] = 50
		grid[int(self.ball.getPos()[0]) + self.length, int(self.ball.getPos()[1]) + abs(min(self.borders))] = 100
		plt.imsave(f"images/image{self.timestep}.png", grid, cmap='hot')
		plt.show(block=False)

	def get_observation(self, agent):
		if agent == "left":
			return np.array([
				self.left_paddle.getPos()[0],
				self.left_paddle.getPos()[1],
				self.right_paddle.getPos()[0],
				self.right_paddle.getPos()[1],
				self.ball.getPos()[0],
				self.ball.getPos()[1]
			])
		elif agent == "right":
			return np.array([
				self.right_paddle.getPos()[0] * -1,
				self.right_paddle.getPos()[1],
				self.left_paddle.getPos()[0] * -1,
				self.left_paddle.getPos()[1],
				self.ball.getPos()[0] * -1,
				self.ball.getPos()[1]
			])

	@functools.lru_cache(maxsize=None)
	def observation_space(self, agent):
		return self.observation_spaces[agent]

	@functools.lru_cache(maxsize=None)
	def action_space(self, agent):
		return self.action_spaces[agent]
	

