import gym
import highway_env

env = gym.make("highway-v0")

done = False
obs_list = []
while not done:
    action = [-0.2, 0]
    obs, reward, done, info = env.step(action)
    obs_list.append(reward)
    env.render()
print(obs_list)