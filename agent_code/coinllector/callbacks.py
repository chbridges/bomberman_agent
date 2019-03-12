import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from random import shuffle

from settings import s
from settings import e

actions = ('RIGHT', 'LEFT', 'UP', 'DOWN')
action_space = len(actions)
last_actions = []

# Hyperparameters in [0,1]
alpha = 0.8     # learning rate
gamma = 0.8     # discount factor
epsilon = 0.0   # randomness in policy, set to >= 1 when reset == 1

# 1: reset weights (overwrite on HDD), 0: use saved weights
reset = 0

if reset == 1:
    epsilon = 1

# Regression
observations = np.zeros((0,5))
rewards = np.zeros((0, action_space))
Q = np.zeros((0, action_space))
reward_highscore = np.NINF
#regr = RandomForestRegressor(n_estimators=100)
#regr = GradientBoostingRegressor(loss='ls')
regr = MultiOutputRegressor(GradientBoostingRegressor(loss='ls'))


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of closest target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    return best


def find_best_action(observation, logger=None):
    global regr, actions
    current_rewards = regr.predict(np.asarray(observation).reshape(1, -1))
    if logger:
        logger.info(np.asarray(observation).reshape(1, -1))
        logger.debug(current_rewards)
    return np.argmax(current_rewards)


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    if reset == 0:
        self.logger.debug('Loading weights')
        global observations, rewards, regr, reward_highscore, Q, action_space
        observations = np.load('observations.npy')
        rewards = np.load('rewards.npy')
        reward_highscore = np.sum(rewards)
        Q = np.load('Q.npy')
        self.logger.debug('Weights loaded. Training regression model...')
        regr.fit(observations, Q)
        self.logger.debug('Model trained')
        self.logger.info('Previous highscore: {0}'.format(reward_highscore))
        observations = np.zeros((0,5))
        rewards = np.zeros((0, action_space))
        Q = np.zeros((0, action_space))
    np.random.seed()


def act(self):
    """Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via self.game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    Set the action you wish to perform by assigning the relevant string to
    self.next_action. You can assign to this variable multiple times during
    your computations. If this method takes longer than the time limit specified
    in settings.py, execution is interrupted by the game and the current value
    of self.next_action will be used. The default value is 'WAIT'.
    """

    global epsilon, observations, reg, actions, last_action

    #self.logger.info('Observing the state')

    # Gather information about the game state
    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    bombs = self.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
    coins = self.game_state['coins']
    crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]

    # Find absolute coordinates of the most immediately interesting targets
    free_space = arena == 0
    targets = []

    #targets.append(look_for_targets(free_space, (x,y), coins, self.logger))
    targets.append(look_for_targets(free_space, (x,y), coins))

    countdown = 0
    for (x,y,t) in bombs:
        if (x,y) == targets[-1]:
            countdown = t;
            break;

    # observation = [agent, relative next coin, action]
    observation = [x,y]

    # Find relative coordinates or (0,0) if target non-existent
    for target in targets:
        if target != None:
            target = (target[0]-x, target[1]-y)
        else:
            target = (0,0)

        observation.append(target[0])
        observation.append(target[1])

    #self.logger.debug(observation)

    # wait, right, left, up, down, bomb

    observation.append(0)
    if np.random.random_sample() < epsilon:
        last_actions.append(np.random.randint(len(actions)))
    else:
        last_actions.append(find_best_action(observation, self.logger))
    self.next_action = actions[last_actions[-1]]

    observations = np.vstack((observations, np.asarray(observation)))


def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """
    #self.logger.debug(f'Encountered {len(self.events)} game event(s)')

    global rewards, gamma, alpha, observations, regr, action_space, last_actions

    if self.game_state['step'] == 2:
        rewards = np.vstack((rewards, np.zeros((0,action_space))))

    reward = 0

    for event in self.events:
        if event == e.INVALID_ACTION:
            reward = reward - 5
        elif event == e.CRATE_DESTROYED:
            reward = reward + 10
        elif event == e.COIN_COLLECTED:
            reward = reward + 50
        elif event == e.KILLED_OPPONENT:
            reward = reward + 10
        elif event == e.KILLED_SELF:
            reward = reward - 10
        #elif event == e.SURVIVED_ROUND:
        #    reward = reward + 100
        else:
            reward = reward - 1

    current_reward = np.zeros((1, action_space))
    current_reward[0][last_actions[-1]] = reward
    self.logger.debug(current_reward)

    #self.logger.debug('Reward: {0}'.format(reward))

    #reward = np.power(gamma, self.game_state['step']) * reward
    '''
    if self.game_state['step'] > 2:
        if reset == 0:
            new_state_optimum = observations[-1]
            new_state_optimum[-1] = find_best_action(observations[-1])
            new_state_reward = regr.predict(np.asarray(new_state_optimum).reshape(1, -1))
            old_state_reward = regr.predict(observations[-2].reshape(1, -1))
            rewards[-1] = old_state_reward + alpha * (rewards[-1] + gamma * new_state_reward - old_state_reward)
    '''

    rewards = np.vstack((rewards, current_reward))

def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    #self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')
    global regr, observations, rewards, reward_highscore, alpha, gamma, Q, reset, last_actions, action_space
    #self.logger.debug('Score: {0}'.format(rewards[0]))

    # Learning
    if reset == 0:
        for i in range(len(rewards)):
            if i < len(rewards)-1:
                current_Q = np.max(regr.predict(observations[i].reshape(1, -1)))
                next_Q = np.max(regr.predict(observations[i+1].reshape(1, -1)))
                current_reward = rewards[i][last_actions[i]]
                Q_s_a = np.zeros((1, action_space))
                Q_s_a[0][last_actions[i]] = current_Q + alpha * (current_reward + gamma * next_Q - current_Q)
                Q = np.vstack((Q, Q_s_a))
            else:
                Q = np.vstack((Q, rewards[i]))
    else:
        Q = rewards
        reset = 0

    Q = np.vstack((Q, np.zeros((1, action_space))))
    regr.fit(observations, Q)

    total = np.sum(rewards)
    self.logger.debug('Return: {0}'.format(total))
    self.logger.debug('Previous highest return: {0}'.format(reward_highscore))
    if total > reward_highscore:
        reward_highscore = total

    np.save('observations.npy', observations)
    np.save('rewards.npy', rewards)
    np.save('Q.npy', Q)
    observations = np.zeros((0,5))
    rewards = np.zeros((0, action_space))
    Q = np.zeros((0, action_space))
    #self.logger.debug(regr.feature_importances_)
