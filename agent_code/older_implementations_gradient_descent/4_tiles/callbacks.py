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
        self.model = np.zeros((6,499))
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

    self.init_epsilon = 0.5
    self.min_epsilon = 0.02
    self.epsilon_decay_ratio = 0.9
    self.n_episodes = 3000
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
    feat = []
    x,y = game_state['self'][3]
    xi = np.array([x-4,x-3,x-2,x-1,x,x+1,x+2,x+3,x+4])
    yi = np.array([y-4,y-3,y-2,y-1,y,y+1,y+2,y+3,y+4])
    c = np.meshgrid(xi,yi)
    xii = c[0].flatten()
    yii = c[1].flatten()
    grid = list(zip(xii,yii))
    bomb_pos = [i[0] for i in game_state['bombs']]
    opp_pos = [i[3] for i in game_state['others']]

    for i in grid:
        xc, yc = i
        tile_arr = [0,0,0,0,0,0]
        if xc<0 or xc>16 or yc<0 or yc >16: tile_arr[0]=1
        else:
            #wall
            if game_state['field'][xc,yc] == -1 : tile_arr[0] = 1
            #crate
            if game_state['field'][xc, yc] == 1: tile_arr[1] = 1
            #bomb
            if i in bomb_pos: tile_arr[2] = 1
            #coin
            if i in game_state['coins']: tile_arr[3] = 1
            #opp
            if i in opp_pos: tile_arr[4] = 1
            #exp
            if game_state['explosion_map'][xc,yc]!=0: tile_arr[5] = 1
        feat.extend(tile_arr)
    pos = np.where(game_state['field'] == 0)
    crate_arr = [0,0,0,0]
    crates = list(zip(pos[0],pos[1]))
    if len(crates) != 0:
        pos_diff_crates = [tuple(map(sum, zip(cur, (-x,-y)))) for cur in crates]
        crate_dist = [sum([abs(ele) for ele in sub]) for sub in pos_diff_crates]
        nearest_crate = crate_dist.index(min(crate_dist))
        crate_dir = np.sign(np.array(crates[nearest_crate]) - np.array(game_state['self'][3]))
        if crate_dir[0] ==1: crate_arr[0] =1
        if crate_dir[0] == -1: crate_arr[1] = 1
        if crate_dir[1] == 1: crate_arr[2] = 1
        if crate_dir[1] == -1: crate_arr[3] = 1
    coin_arr = [0,0,0,0]
    if len(game_state['coins']) != 0:
        pos_diff_coins = [tuple(map(sum, zip(cur, (-x,-y)))) for cur in game_state['coins']]
        coins_dist = [sum([abs(ele) for ele in sub]) for sub in pos_diff_coins]
        nearest_coin = coins_dist.index(min(coins_dist))
        coin_dir = np.sign(np.array(crates[nearest_coin]) - np.array(game_state['self'][3]))
        if coin_dir[0] ==1: coin_arr[0] =1
        if coin_dir[0] == -1: coin_arr[1] = 1
        if coin_dir[1] == 1: coin_arr[2] = 1
        if coin_dir[1] == -1: coin_arr[3] = 1
    opp_arr = [0,0,0,0]
    if len(opp_pos) != 0:
        pos_diff_opp = [tuple(map(sum, zip(cur, (-x, -y)))) for cur in opp_pos]
        opp_dist = [sum([abs(ele) for ele in sub]) for sub in pos_diff_opp]
        nearest_opp = opp_dist.index(min(opp_dist))
        opp_dir = np.sign(np.array(crates[nearest_opp]) - np.array(game_state['self'][3]))
        if opp_dir[0] == 1: opp_arr[0] = 1
        if opp_dir[0] == -1: opp_arr[1] = 1
        if opp_dir[1] == 1: opp_arr[2] = 1
        if opp_dir[1] == -1: opp_arr[3] = 1
    feat.extend(crate_arr)
    feat.extend(opp_arr)
    feat.extend(coin_arr)
    if short_escape(game_state): feat.append(1)
    else: feat.append(0)
    return np.array(feat)

def short_escape(game_state):
    x, y = game_state['self'][3]
    esc = False
    field = game_state['field']
    if field[x+1,y] == 0 and field[x+1,y+1] ==0: esc =True
    if field[x + 1, y] == 0 and field[x + 1, y - 1] == 0: esc = True
    if field[x -1, y] == 0 and field[x - 1, y + 1] == 0: esc = True
    if field[x - 1, y] == 0 and field[x - 1, y - 1] == 0: esc = True
    if field[x,y+1] == 0 and field[x+1,y+1] == 0: esc = True
    if field[x, y + 1] == 0 and field[x - 1, y + 1] == 0: esc = True
    if field[x, y - 1] == 0 and field[x + 1, y - 1] == 0: esc = True
    if field[x, y - 1] == 0 and field[x - 1, y - 1] == 0: esc = True
    return esc







