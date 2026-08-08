from wrappers import *
import random

env = preProcess(make_env(render_mode="human"))
episodes = 5
for episode in episodes:
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = random.choice([0,1,2,3,4,5])
        n_state, reward, done, info = env.step(action)
        score += reward
    print("Episode: {}, Score: {}".format(episode, score))
env.close()