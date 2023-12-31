from pong import PongGameEnv
import time

# Create the environment
env = PongGameEnv(speed=1, paddle_size=(2, 6), borders=(-50, 50), length=30)

# Get the initial state
state = env.reset()

rewards = {"left": 0, "right": 0}

while True:
    # Choose a random action
    actions = {"left": env.action_space("left").sample(), "right": env.action_space("right").sample()}

    # Perform the action
    obs, reward, terminations, truncations, info = env.step(actions)
    rewards["left"] += reward["left"]
    rewards["right"] += reward["right"]

    if terminations["left"] or terminations["right"]:
        print("Game over!")
        print("Left paddle score: {}".format(rewards["left"]))
        print("Right paddle score: {}".format(rewards["right"]))
        rewards = {"left": 0, "right": 0}
        env.reset()