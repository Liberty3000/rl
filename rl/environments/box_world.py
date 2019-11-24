import random, gym
from gym.spaces.discrete import Discrete
from gym.spaces import Box
import numpy as np, matplotlib.pyplot as mp

agent_color = [128, 128, 128]

action_space = {
    0: (-1, 0),
    1: (1,  0),
    2: (0, -1),
    3: (0,  1)
}

goal_color = [255, 255, 255]
grid_color = [220, 220, 220]

colors = {0: [  0,   0,   0],
          1: [230, 190, 255],
          2: [170, 255, 195],
          3: [255, 250, 200],
          4: [255, 216, 177],
          5: [250, 190, 190],
          6: [240,  50, 230],
          7: [145,  30, 180],
          8: [ 67,  99, 216],
          9: [ 66, 212, 244],
          10:[ 60, 180,  75],
          11:[191, 239,  69],
          12:[255, 255,  25],
          13:[245, 130,  49],
          14:[230,  25,  75],
          15:[128,   0,   0],
          16:[154,  99,  36],
          17:[128, 128,   0],
          18:[ 70, 153, 144],
          19:[  0,   0, 117]}

def sampling_pairs(pairs, n=12):
    possibilities = set(range(1, n*(n-1)))
    keys,locks = [],[]
    for k in range(pairs):
        key = random.sample(possibilities, 1)[0]
        key_x, key_y = key//(n-1), key%(n-1)

        lock_x, lock_y = key_x, key_y + 1

        to_remove = [key_x * (n-1) + key_y] + \
                    [key_x * (n-1) + i + key_y for i in range(1, min(2, n - 2 - key_y) + 1)] + \
                    [key_x * (n-1) - i + key_y for i in range(1, min(2, key_y) + 1)]

        possibilities -= set(to_remove)

        keys.append([key_x, key_y])
        locks.append([lock_x, lock_y])

    agent_xy = random.sample(possibilities, 1)
    possibilities -= set(agent_xy)
    first_key = random.sample(possibilities, 1)

    agent_xy = np.array([agent_xy[0]//(n-1), agent_xy[0]%(n-1)])
    first_key = first_key[0]//(n-1), first_key[0]%(n-1)
    return keys, locks, first_key, agent_xy

def generate(n=12, goal_length=3, num_distractor=2, distractor_length=2, seed=None, verbose=False):
    if seed is not None: random.seed(seed)

    world_dic = {}
    world = np.ones((n, n, 3)) * 220
    goal_colors = random.sample(range(len(colors)), goal_length - 1)

    distractor_possible_colors = [color for color in range(len(colors)) if color not in goal_colors]
    distractor_colors = [random.sample(distractor_possible_colors, distractor_length) for k in range(num_distractor)]
    distractor_roots  = random.choices(range(goal_length - 1), k=num_distractor)
    keys, locks, first_key, agent_xy = sampling_pairs(goal_length - 1 + distractor_length * num_distractor, n)

    # create the goal path
    for i in range(1, goal_length):
        if i == goal_length - 1: color = goal_color  # the final key is white
        else: color = colors[goal_colors[i]]

        banner = 'placed a key with color {} on position {}, corresponding lock at {} with color {}'
        if verbose: print(banner.format(color, keys[i-1], locks[i-1], colors[goal_colors[i-1]]))
        world[ keys[i-1][0],  keys[i-1][1]] = np.array(color)
        world[locks[i-1][0], locks[i-1][1]] = np.array(colors[goal_colors[i-1]])

    # keys[0] is an orphand key, so skip it
    world[first_key[0], first_key[1]] = np.array(colors[goal_colors[0]])
    banner = 'placed the first key with color {} on position {}'
    if verbose: print(banner.format(goal_colors[0], first_key))

    # place distractors
    for i, (distractor_color, root) in enumerate(zip(distractor_colors, distractor_roots)):
        key_distractor = keys[goal_length-1 + i*distractor_length: goal_length-1 + (i+1)*distractor_length]
        color_lock = colors[goal_colors[root]]
        color_key = colors[distractor_color[0]]
        world[key_distractor[0][0], key_distractor[0][1] + 1] = np.array(color_lock)
        world[key_distractor[0][0], key_distractor[0][1]] = np.array(color_key)
        for k, key in enumerate(key_distractor[1:]):
            color_lock = colors[distractor_color[k-1]]
            color_key = colors[distractor_color[k]]
            world[key[0], key[1]] = np.array(color_key)
            world[key[0], key[1]+1] = np.array(color_lock)

    # place the agent
    world[agent_xy[0], agent_xy[1]] = np.array(agent_color)
    return world, keys, locks, first_key, agent_xy

def update_color(world, previous_agent_loc, new_agent_loc):
        world[previous_agent_loc[0], previous_agent_loc[1]] = grid_color
        world[new_agent_loc[0], new_agent_loc[1]] = agent_color

def is_empty(room):
    return np.array_equal(room, grid_color) or np.array_equal(room, agent_color)

class BoxWorld(gym.Env):
    def __init__(self, n, goal_length, num_distractor, distractor_length, max_steps=2**10):
        self.goal_length = goal_length
        self.num_distractor = num_distractor
        self.distractor_length = distractor_length
        self.n = n
        self.pairs = goal_length - 1 + distractor_length * num_distractor

        self.step_cost  = 1e-1
        self.reward_gem = 10
        self.reward_key = 0

        self.max_steps = max_steps
        self.action_space = Discrete(len(action_space))
        self.observation_space = Box(low=0, high=255, shape=(n, n, 3), dtype=np.uint8)

        self.owned_key = grid_color

        self.reset()

    def save(self):
        np.save('box-world_{}.npy'.format(str(uuid.uuid4())[:8]))

    def step(self, action, verbose=False):
        self.num_env_steps += 1

        change = action_space[action]
        new_position = self.agent_xy + change
        current_position = self.agent_xy.copy()

        reward = -self.step_cost
        terminal = self.num_env_steps == self.max_steps

        if np.any(new_position < 0) or np.any(new_position >= self.n):
            possible_move = False
        elif np.array_equal(new_position, [0, 0]):
            possible_move = False
        elif is_empty(self.world[new_position[0], new_position[1]]):
            # no key, no lock
            possible_move = True
        elif new_position[1] == 0 or is_empty(self.world[new_position[0], new_position[1]-1]):
            # it's a key
            if is_empty(self.world[new_position[0], new_position[1]+1]):
                # key is not locked
                possible_move = True
                self.owned_key = self.world[new_position[0], new_position[1]].copy()
                self.world[0,0] = self.owned_key
                if np.array_equal(self.world[new_position[0], new_position[1]], goal_color):
                    # goal reached
                    reward += self.reward_gem
                    terminal = True
                else:
                    reward += self.reward_key
            else:
                possible_move = False
        else:
            # it's a lock
            if np.array_equal(self.world[new_position[0], new_position[1]], self.owned_key):
                # the lock matches the key
                possible_move = True
            else:
                possible_move = False
                banner = 'lock color is {}, but owned key is {}'
                if verbose: print(banner.format(self.world[new_position[0], new_position[1]], self.owned_key))

        if possible_move:
            self.agent_xy = new_position
            update_color(self.world, previous_agent_loc=current_position, new_agent_loc=new_position)

        metadata = {}
        return self.world, reward, terminal, metadata

    def reset(self, seed=None):
        args = generate(n=self.n, goal_length=self.goal_length,
                        num_distractor=self.num_distractor,
                        distractor_length=self.distractor_length,
                        seed=seed)

        self.world, self.keys, self.locks, self.first_key, self.agent_xy = args

        self.num_env_steps = 0
        return self.world

    def render(self, mode='human'):
        img = self.world.astype(np.uint8)
        if mode == 'human':
            mp.imshow(img, vmin=0, vmax=255, interpolation='none')
            mp.show()
        else: return img

    def extract_objects(self, feature_maps):
        objects = [list(self.agent_xy)] + self.keys + self.locks
        return th.stack([feature_maps[...,x,y] for x,y in objects], dim=1)
