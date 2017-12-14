import gym
env = gym.make('Breakout-v0')
# or 'SpaceInvaders-v0' or 'MsPacman-v0'
env.reset()
for _ in range(1000):
    env.render()
	env.step(env.action_space.sample())
