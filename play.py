from wrappers import *
import random
import time

WATCH = False #allows agent to be watched

env = preProcess(make_env(render_mode="human" if WATCH else None))

actions = env.action_space.n
episodes = 5

for episode in range(episodes):

    state, info = env.reset()
    done = False
    score = 0

    while not done:
        action = random.randrange(actions)
        n_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        score += reward

        if WATCH:
            time.sleep(0.02)

    print(f"Episode: {episode + 1}, Score: {score}")

env.close()