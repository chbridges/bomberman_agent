import numpy as np

rewards = np.load('rewards.npy')
last_actions = np.zeros((0,1), dtype=np.int16)
for i in range(len(rewards)):
    action = np.nonzero(rewards[i])[0][0]
    last_actions = np.vstack((last_actions, [action]))
print(rewards.shape, last_actions.shape)
np.save('last_actions.npy', last_actions)
