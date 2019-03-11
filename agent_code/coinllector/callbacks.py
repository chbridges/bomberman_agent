import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from random import shuffle

from settings import s
from settings import e

# Hyperparameters in [0,1]
alpha = 0.8     # learning rate
gamma = 0.95    # discount factor
epsilon = 0.2   # randomness in policy, set to >= 1 when reset == 1

# 1: reset weights (overwrite on HDD), 0: use saved weights
reset = 0

# Regression
observations = np.zeros((0,3))
rewards = np.zeros((0,1))
reward_highscore = np.NINF
regr = RandomForestRegressor(n_estimators=100)
#regr = GradientBoostingRegressor(loss='ls')

actions = ('RIGHT', 'LEFT', 'UP', 'DOWN')


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
    max_reward = np.NINF
    max_action = 0
    for i in range(len(actions)):
        observation[-1] = i
        current_reward = regr.predict(np.asarray(observation).reshape(1, -1))
        if logger:
            self.logger.info(np.asarray(observation).reshape(1, -1))
            self.logger.debug('{0}: Reward {1}'.format(actions[i], current_reward))
        if current_reward > max_reward:
            max_action = i
            max_reward = current_reward
    return max_action


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
        global observations, rewards, reg, reward_highscore
        observations = np.load('observations.npy')
        rewards = np.load('rewards.npy')
        reward_highscore = np.mean(rewards)
        self.logger.debug('Weights loaded. Training regression model...')
        global reg, alpha, gamma, epsilon
        regr.fit(observations, rewards.ravel())
        self.logger.debug('Model trained')
        self.logger.info('Previous highscore: {0}'.format(reward_highscore))
        observations = np.zeros((0,3))
        rewards = np.zeros((0,1))
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

    global epsilon, observations, reg, actions

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

    # observation = [relative next coin, action]
    observation = []

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
        observation[-1] = np.random.randint(len(actions))
    else:
        observation[-1] = find_best_action(observation)
    self.next_action = actions[observation[-1]]

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

    global rewards, gamma, alpha, observations

    if self.game_state['step'] == 2:
        rewards = np.vstack((rewards, 0))

    reward = 0

    for event in self.events:
        if event == e.INVALID_ACTION:
            reward = reward - 10
        elif event == e.CRATE_DESTROYED:
            reward = reward + 10
        elif event == e.COIN_COLLECTED:
            reward = reward + 100
        elif event == e.KILLED_OPPONENT:
            reward = reward + 100
        elif event == e.KILLED_SELF:
            reward = reward - 100
        elif event == e.SURVIVED_ROUND:
            reward = reward + 100

    #self.logger.debug('Reward: {0}'.format(reward))

    reward = np.power(gamma, self.game_state['step']) * reward

    if reset == 0:
        new_state_optimum = observations[-1]
        new_state_optimum[-1] = find_best_action(observations[-1])
        new_state_reward = regr.predict(np.asarray(new_state_optimum).reshape(1, -1))
        new_state_reward = np.power(gamma, self.game_state['step']+1) * new_state_reward
        rewards[-1] = (1-alpha)*rewards[-1] + alpha*new_state_reward

    rewards = np.vstack((rewards, reward))

def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    #self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')
    global reg, observations, rewards, reward_highscore, alpha
    #self.logger.debug('Score: {0}'.format(rewards[0]))
    total = np.mean(rewards)
    self.logger.debug('Mean reward: {0}'.format(total))
    self.logger.debug('Previous highest mean: {0}'.format(reward_highscore))
    if total > reward_highscore:
        reward_highscore = total
    regr.fit(observations, rewards.ravel())
    np.save('observations.npy', observations)
    np.save('rewards.npy', rewards)
    observations = np.zeros((0,3))
    rewards = np.zeros((0,1))
    self.logger.debug(regr.feature_importances_)
