# bomberman_agent
A reinforcement learning agent for the Bomberman framework found at https://github.com/ukoethe/bomberman_rl
using Q-learning with gradient boosted trees and an ε-greedy exploration-exploitation strategy.

The folder agent_code currently contains two version of the agent:
1. coinllector is coin collecting agent, developed for an arena without crates or enemies
2. cratemate is an advanced agent, able to place bombs in order to destroy crates

The first agent furthermore contains a backup subfolder including a good sample dataset,
the recorded logs across ten test games and the corresponding replays.
