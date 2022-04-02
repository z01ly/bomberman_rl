import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_DICT = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}


def setup(self):
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        # self.model is the weights, dimension: 1 x feature_num
        # Now, suppose there are 10 features
        self.model = np.zeros((6,2313))
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

    self.init_epsilon = 1.0
    self.min_epsilon = 0.1
    self.epsilon_decay_ratio = 0.9
    self.n_episodes = 10000
    self.epsilons = decay_schedule(self, self.init_epsilon, self.min_epsilon, self.epsilon_decay_ratio, self.n_episodes)

    #print(self.model)


def decay_schedule(self, init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps

    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value

    values = np.pad(values, (0, rem_steps), 'edge')

    return values


def act(self, game_state: dict) -> str:
    cur_state = state_to_features(game_state)
    val_ac = valid_actions(game_state)
    self.logger.debug(val_ac)

    # epsilon-greedy
    # random_prob = 0.2
    round_n = game_state['round'] - 1
    random_prob = self.epsilons[round_n]

    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        prob = np.ones(6)/6
        if len(val_ac) > 0:
            prob = np.zeros(6)
            counter = 0
            for i in range(6):
                if i in val_ac and i <4: counter+=2
                if i in val_ac and i>=4: counter+=1
            for i in val_ac:
                if i<4:prob[i] = 2/counter
                else: prob[i] = 1/counter
        return np.random.choice(ACTIONS, p=prob)

    self.logger.debug("Querying model for action.")
    qvals = self.model @ cur_state
    self.logger.debug(qvals)

    #print(qvals)

    qvals = np.squeeze(qvals)
    invalid_ac = list(set(range(6)) - set(val_ac))
    qvals[invalid_ac] = - np.inf
    '''
    a_list = np.argwhere(qvals == np.max(qvals)).flatten().tolist()
    if len(a_list) == 1:
        a = a_list[0]
    else:
        self.logger.debug(qvals)
        self.logger.debug(a_list)
        a = random.choice(a_list)
    '''
    a = np.argmax(qvals)
    # print(ACTIONS[a])

    return ACTIONS[a]


def valid_actions(game_state):
    val_ac = []
    x,y = game_state['self'][3]
    bomb_pos = [i[0] for i in game_state['bombs']]
    opp_pos = [i[3] for i in game_state['others']]
    #for movement free,bombs,others,expl
    #up
    if (x,y-1) not in bomb_pos and (x,y-1) not in opp_pos and game_state['explosion_map'][x,y-1]==0 and game_state['field'][x,y-1] == 0:
        val_ac.append(0)
    #right
    if (x+1,y) not in bomb_pos and (x+1,y) not in opp_pos and game_state['explosion_map'][x+1,y]==0 and game_state['field'][x+1,y] == 0:
        val_ac.append(1)
    #down
    if (x,y+1) not in bomb_pos and (x,y+1) not in opp_pos and game_state['explosion_map'][x,y+1]==0 and game_state['field'][x,y+1] == 0:
        val_ac.append(2)
    #left
    if (x-1,y) not in bomb_pos and (x-1,y) not in opp_pos and game_state['explosion_map'][x-1,y]==0 and game_state['field'][x-1,y] == 0:
        val_ac.append(3)
    #wait
    if (x,y) not in bomb_pos and game_state['explosion_map'][x,y]==0:
        val_ac.append(4)
    #bomb
    if game_state['self'][2]: val_ac.append(5)

    return val_ac


def state_to_features(game_state: dict) -> np.array:
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    feat = np.zeros((289,8))
    flat_field = game_state['field'].flatten()
    #wall 0
    walls = np.where(flat_field == -1)[0]
    feat[walls,0] =1
    #crate 1
    crates = np.where(flat_field == 1)[0]
    feat[crates, 1] = 1
    #self 2
    xo,yo = game_state['self'][3]
    feat[17*xo+yo,2] = 1
    #other 3
    for i in game_state['others']:
        xi,yi = i[3]
        feat[17*xi+yi,3] = 1
    #coin 4
    for i in game_state['coins']:
        xi,yi = i
        feat[17*xi+yi,4] = 1
    #bomb 5
    for i in game_state['bombs']:
        xi,yi = i[0]
        feat[17 * xi+ yi, 5] = 1
    #expl 6
    flat_exp = game_state['explosion_map'].flatten()
    expl = np.where(flat_exp != 0)[0]
    feat[expl,6] = 1
    #threat 7
    for i in game_state['bombs']:
        xi,yi = i[0]
        if xi%2 == 1:
            xit = np.array([xi-3,xi-2,xi-1,xi,xi+1,xi+2,xi+3])
            xit = np.clip(xit, 1, 15)
            for ii in xit:
                feat[ii*17+yi,7] = 1
        if yi%2 == 1:
            yit = np.array([yi-3,yi-2,yi-1,yi,yi+1,yi+2,yi+3])
            yit = np.clip(yit, 1, 15)
            for ii in yit:
                feat[xi * 17 + ii, 7] = 1
    feat_flat = feat.flatten()
    feat_flat = np.append(feat_flat,game_state['step'])
    return feat_flat
