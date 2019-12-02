import gym, random
from collections import defaultdict
import matplotlib.pyplot as mp, numpy as np

entities = {
    'carrier':   5,
    'battleship':4,
    'submarine': 3,
    'destroyer': 3,
    'cruiser':   2
}

grid_size = (10,10)

stochastic_policy = lambda:np.random.randint(np.prod(grid_size))

class Battleship(gym.Env):
    def __init__(self, opponent_type='stochastic', fool_me_twice=False):
        self.opponent_type = opponent_type
        self.fool_me_twice = fool_me_twice
        self.reset()

    def reset(self):
        self.timer = 0
        self.action_space = grid_size
        self.n_action = np.prod(grid_size)
        self.actions = range(self.n_action)
        self.metadata = dict(legal_actions=self.actions)

        self.friends, self.positives = self.initialize_entities(grid_size)
        self.friends_init = self.friends.copy()
        self.friends_attack_history = []
        self.friends_action = None

        self.enemies, self.negatives = self.initialize_entities(grid_size)
        self.enemies_init = self.enemies.copy()
        self.enemies_attack_history = []
        self.enemies_action = None
        return self.encode_observation(self.friends_attack_history, self.negatives)

    def initialize_entities(self, grid_size, verbose=False):
        grid = np.zeros(grid_size)
        catalog = defaultdict(list)
        for ship,length in entities.items():
            while True:
                idx = np.random.randint(np.prod(grid_size))
                x,y = np.unravel_index(idx, grid_size)
                if grid[x,y] == 1: continue
                banner = 'attempting to place {} with root {}'
                if verbose: print(banner.format(ship,(x,y)))

                ax = random.choice([0,1])
                coords = []
                for i in range(length):
                    if ax:xy = [x+i,y]
                    else: xy = [x,y+i]

                    coords.append(xy)
                coords = np.asarray(coords)

                banner = '{} candidate locations: {}'
                if verbose: print(banner.format(ship,coords))
                try:
                    sentinel = (grid[coords].all() == 0)
                    if sentinel:
                        for (x,y) in coords:
                            grid[x,y] = entities[ship]
                            catalog[ship].append((x,y))
                        banner = 'successfully placed {}\n'
                        if verbose: print(banner.format(ship))
                        break
                except: continue
        return grid, catalog

    def encode_observation(self, attacks=[], catalog=None):
        history = np.zeros(grid_size)
        for xy in attacks:
            targets = [ij for coords in catalog.values() for ij in coords]
            if xy in targets: history[xy] = 1
            else: history[xy] = -1
        return history

    def calculate_damage(self, action, grid, entities, attacks):
        terminal = False

        if action in attacks:
            label,reward = 'MISS ⚠️',-1
        else:
            if grid[action] != 0 and action not in attacks:
                label,reward = 'BANG 💥',1
            if grid[action] == 0:
                label,reward = 'MISS 💦',0

            mask = np.ones(grid_size)
            mask[action] = 0
            grid *= mask
            terminal = not np.any(grid)

        return grid, np.asarray(reward), np.asarray(int(terminal)), label

    def sunken(self, coords, attacks):
        return all([coord in attacks for coord in coords])

    def legal_actions(self, history):
        actions = range(np.prod(grid_size))
        return [a for a in actions if not np.unravel_index(a, grid_size) in history]

    def step(self, action, verbose=False):
        self.friends_action = np.unravel_index(action, grid_size)
        self.enemies, reward, terminal, label = self.calculate_damage(
        self.friends_action, self.enemies, self.negatives, self.friends_attack_history)
        self.friends_attack_history.append(self.friends_action)
        self.metadata['legal_actions'] = self.legal_actions(self.friends_attack_history)
        if self.fool_me_twice: self.metadata['legal_actions'] = self.actions
        self.metadata['friends_attack'] = label
        if terminal:
            self.metadata['status'] = 'success'
            return self.enemies, reward, terminal, self.metadata

        if self.opponent_type == 'stochastic':
            output = np.random.choice(self.metadata['enemies_legal_actions'])
        if self.opponent_type == 'linear':
            output = self.timer
        if self.opponent_type == 'heuristic':
            output = stochastic_policy()

        self.enemies_action = np.unravel_index(output, grid_size)
        self.friends,penalty, terminal, label = self.calculate_damage(
        self.enemies_action, self.friends, self.positives, self.enemies_attack_history)
        self.enemies_attack_history.append(self.enemies_action)
        self.metadata['enemies_legal_actions'] = self.legal_actions(self.enemies_attack_history)
        if self.fool_me_twice: self.metadata['enemies_legal_actions'] = self.actions
        self.metadata['enemies_attack'] = label
        if terminal:
            self.metadata['status'] = 'failure'
            return self.enemies, reward, terminal, self.metadata


        self.metadata['status'] = 'neutral'
        obs = self.encode_observation(self.friends_attack_history, self.negatives)
        self.timer += 1
        return obs, reward, terminal, self.metadata

    def render(self, verbose=False):
        banner = '\n\t{}player {} @ {} | {}\n'
        statuses = {False:'AFLOAT',True:'SUNKEN'}
        fig,((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8)) = mp.subplots(2,4, figsize=(17,7))

        if verbose:
            if len(self.friends_attack_history) > 0:
                print('_' * 80, banner.format('🔵', 1, self.friends_attack_history[-1], self.metadata['friends_attack']))
                for k,v in self.negatives.items():
                    status = statuses[self.sunken(v, self.friends_attack_history)]
                    print('{:10}|{}|'.format(k, status), v)
                print('_' * 80)
            if len(self.enemies_attack_history) > 0:
                print('_' * 80, banner.format('🔴', 2, self.enemies_attack_history[-1], self.metadata['enemies_attack']))
                for k,v in self.positives.items():
                    status = statuses[self.sunken(v, self.enemies_attack_history)]
                    print('{:10}|{}|'.format(k, status), v)
                print('_' * 80)

        ax1.set_title('Where We Attacked')
        ax2.set_title('What We Know')
        ax3.set_title('Where They Are')
        ax4.set_title('Where They Began')

        ax5.set_title('Where They Attacked')
        ax6.set_title('What They Know')
        ax7.set_title('Where We Are')
        ax8.set_title('Where We Began')

        friends_obs = self.encode_observation(self.friends_attack_history, self.negatives)
        friends_recent = np.zeros(grid_size)
        friends_recent[self.friends_action] = 1
        ax1.imshow(friends_recent, cmap='Reds')
        ax2.imshow(friends_obs, cmap='hot')
        ax3.imshow(self.enemies, cmap='OrRd')
        ax4.imshow(self.enemies_init, cmap='OrRd')

        enemies_obs = self.encode_observation(self.enemies_attack_history, self.positives)
        enemies_recent = np.zeros(grid_size)
        enemies_recent[self.enemies_action] = 1
        ax5.imshow(enemies_recent, cmap='Blues')
        ax6.imshow(enemies_obs, cmap='winter')
        ax7.imshow(self.friends, cmap='BuPu')
        ax8.imshow(self.friends_init, cmap='BuPu')

        for i in range(0,10):
            ax1.plot([i+.5]*10,range(10), color='blue')
            ax1.plot(range(10),[i+.5]*10, color='blue')
            ax2.plot([i+.5]*10,range(10), color='red')
            ax2.plot(range(10),[i+.5]*10, color='red')
            ax3.plot([i+.5]*10,range(10), color='black')
            ax3.plot(range(10),[i+.5]*10, color='black')
            ax4.plot([i+.5]*10,range(10), color='black')
            ax4.plot(range(10),[i+.5]*10, color='black')
            ax5.plot([i+.5]*10,range(10), color='red')
            ax5.plot(range(10),[i+.5]*10, color='red')
            ax6.plot([i+.5]*10,range(10), color='black')
            ax6.plot(range(10),[i+.5]*10, color='black')
            ax7.plot([i+.5]*10,range(10), color='blue')
            ax7.plot(range(10),[i+.5]*10, color='blue')
            ax8.plot([i+.5]*10,range(10), color='blue')
            ax8.plot(range(10),[i+.5]*10, color='blue')

        ax1.set_xticks([]),ax1.set_yticks([])
        ax2.set_xticks([]),ax2.set_yticks([])
        ax3.set_xticks([]),ax3.set_yticks([])
        ax4.set_xticks([]),ax4.set_yticks([])
        ax5.set_xticks([]),ax5.set_yticks([])
        ax6.set_xticks([]),ax6.set_yticks([])
        ax7.set_xticks([]),ax7.set_yticks([])
        ax8.set_xticks([]),ax8.set_yticks([])

        mp.tight_layout()
        mp.show()
