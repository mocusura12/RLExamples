
# coding: utf-8

# In[1]:


# from github.com/openai/gym/examples/agents/keyboard_agent.py
# %load keyboard_agent.py
#!/usr/bin/env python
from __future__ import print_function

import sys, gym

#
# Test yourself as a learning agent! Pass environment name as a command-line argument.
#

#envName = 'CartPole-v1'
envName = 'CartPole-v1'

if (envName == 'CartPole-v1'):
    timeSleep = 0.1
if (envName == 'MountainCar-v0'):    
    timeSleep = 0.05


env = gym.make(envName)

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
ROLLOUT_TIME = 2000
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

def rollout(env):
    import time
    
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    for t in range(ROLLOUT_TIME):
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        env.render()

        time.sleep(timeSleep)
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            import time
            env.render()
            time.sleep(0.1)
        if (t % 10 == 0):
            print(t)

print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

while 1:
    rollout(env)

