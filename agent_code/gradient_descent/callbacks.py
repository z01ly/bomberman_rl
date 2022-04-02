import os
import pickle
import random
from collections import deque

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_DICT = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}


def setup(self):
    if self.train and not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        # self.model is the weights, dimension: 1 x feature_num
        # Now, suppose there are 10 features
        self.model = np.zeros((6,86))
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        #print(self.model)


    self.init_epsilon = 0.5
    self.min_epsilon = 0.1
    self.epsilon_decay_ratio = 0.5
    self.n_episodes = 3000
    self.epsilons = decay_schedule(self, self.init_epsilon, self.min_epsilon, self.epsilon_decay_ratio, self.n_episodes)
    self.coordinate_history = deque([], 20)
    self.loop = False
    self.trap = False
    self.last_action = None
    self.cur_action = None
    self.bomb_dropped = []

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
    if game_state['step'] == 1:
        self.coordinate_history = deque([], 20)
        #print(self.coordinate_history)
        self.last_action = None
        self.cur_action = None
        self.bomb_dropped = []
        #print(self.bomb_dropped)
    self.loop = False
    self.trap = False
    cur_state = state_to_features(self.cur_action,game_state)
    val_ac = valid_actions(game_state,self)
    self.logger.debug(val_ac)

    x, y = game_state['self'][3]
    if self.coordinate_history.count((x, y)) > 3:
        self.loop = True
    self.coordinate_history.append((x, y))
    self.logger.debug(self.loop)
    if cannot_escape(game_state,x,y) == 1:
        self.trap = True
    self.logger.debug(self.trap)

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
        a = np.random.choice([0,1,2,3,4,5], p=prob)
    else:
        self.logger.debug("Querying model for action.")
        qvals = self.model @ cur_state
        self.logger.debug(qvals)

        qvals = np.squeeze(qvals)
        invalid_ac = list(set(range(6)) - set(val_ac))
        qvals[invalid_ac] = - np.inf
        a = np.argmax(qvals)

        if a == 5:
            self.bomb_dropped.append((x,y))


        if a == 4 and game_state['self'][2]:
            a = np.random.choice(6, p=[.1, .1, .1, .1, .1, .5])

        if self.loop and game_state['self'][2]:
            a = np.random.choice(6, p=[.1, .1, .1, .1, .1, .5])

    self.last_action = self.cur_action
    self.cur_action = a
    return ACTIONS[a]


def valid_actions(game_state,self):
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
    if game_state['self'][2] and not game_state['step'] == 1 and (x,y) not in self.bomb_dropped:
        val_ac.append(5)

    return val_ac


def state_to_features(last_action, game_state: dict) -> np.array:
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    feat = []
    x,y = game_state['self'][3]
    grid = [(x+1,y),(x-1,y),(x,y+1),(x,y-1),(x+2,y),(x-2,y),(x,y+2),(x,y-2)]
    bomb_pos = [i[0] for i in game_state['bombs']]
    opp_pos = [i[3] for i in game_state['others']]

    for i in grid:
        xc, yc = i
        tile_arr = [0,0,0, 0,0,0,0,0,0]
        if xc <1 or xc>15 or yc < 1 or yc >15:
            tile_arr[0] = 1
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
            #threat
            for i in bomb_pos:
                xb,yb = i
                if xb == xc:
                    if abs(yb-yc) < 4:
                        if xb%2 ==1: tile_arr[6] = 1
                if yb == yc:
                    if abs(xb-xc) < 4:
                        if yb%2 == 1:tile_arr[6] = 1
            #escape
            tile_arr[7] = cannot_escape(game_state,xc,yc)
            ##no straigth line
            neighbours_x = [(xc+1,yc),(xc-1,yc)]
            neighbours_y = [(xc,yc+1),(xc,yc-1)]
            for ii in neighbours_y:
                xcc,ycc = ii
                if game_state['field'][xcc,ycc] == 0:
                    for iii in neighbours_x:
                        xccc,yccc = iii
                        if game_state['field'][xccc,yccc] == 0:
                            tile_arr[8]= 1

        feat.extend(tile_arr)
    pos = np.where(game_state['field'] == 1)
    crates = list(zip(pos[0],pos[1]))
    crate_arr = [0,0]
    if len(crates) != 0:
        pos_diff_crates = [tuple(map(sum, zip(cur, (-x,-y)))) for cur in crates]
        crate_dist = [sum([abs(ele) for ele in sub]) for sub in pos_diff_crates]
        nearest_crate = crate_dist.index(min(crate_dist))
        crate_dir = pos_diff_crates[nearest_crate]
        crate_arr[0] = np.sign(crate_dir[0])*(16 - abs(crate_dir[0]))/8
        crate_arr[1] = np.sign(crate_dir[1])*(16 - abs(crate_dir[1]))/8
    coin_arr = [0,0]
    if len(game_state['coins']) != 0:
        pos_diff_coins = [tuple(map(sum, zip(cur, (-x,-y)))) for cur in game_state['coins']]
        coins_dist = [sum([abs(ele) for ele in sub]) for sub in pos_diff_coins]
        nearest_coin = coins_dist.index(min(coins_dist))
        coin_dir = pos_diff_coins[nearest_coin]
        coin_arr[0] = np.sign(coin_dir[0])*(16 - abs(coin_dir[0]))/8
        coin_arr[1] = np.sign(coin_dir[1])*(16 - abs(coin_dir[1]))/8
    feat.extend(crate_arr)
    feat.extend(coin_arr)
    bomb_dir = [0,0,0,0]
    for bomb in bomb_pos:
        xb,yb = bomb
        if xb == x:
            if -4 < yb-y < 0:
                if x%2 == 1:bomb_dir[0] = 1
            if 0 < yb-y < 4:
                if x%2 == 1:bomb_dir [1] = 1
        if yb == y:
            if -4 < xb-x < 0:
                if y%2 == 1:bomb_dir[2] = 1
            if 0 < xb-x < 4:
                if y%2 == 1:bomb_dir [3] = 1
    feat.extend(bomb_dir)
    last_ac = [0,0,0,0,0,0]
    if last_action != None:
        last_ac[last_action] = 1
    feat.extend(last_ac)
    return np.array(feat)



def cannot_escape(game_state,x,y):
    field = game_state['field']
    bombs = [i[0] for i in game_state['bombs']]
    vec = 0
    if len(bombs) == 0:
        return vec

    dead_ends = trap_area(field)
    trap1 = in_trap_1(field, bombs, dead_ends, x, y)
    trap2 = in_trap_2(field, bombs, dead_ends, x, y)
    trap3 = in_trap_3(field, bombs, dead_ends, x, y)
    if (trap1 == True) or (trap2 == True) or (trap3 == True)  :
        vec = 1

    return vec
def trap_area(field):
    cols = range(1, field.shape[0] - 1)
    rows = range(1, field.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (field[x, y] == 0) \
    and ([field[x + 1, y], field[x - 1, y], field[x, y + 1], field[x, y - 1]].count(0) == 1)]

    return dead_ends


def in_trap_1(field, bombs, dead_ends, x, y):
    if ((x, y + 1) in bombs) and ((x, y) in dead_ends):
        return True
    if ((x - 1, y) in bombs) and ((x, y) in dead_ends):
        return True
    if ((x, y - 1) in bombs) and ((x, y) in dead_ends):
        return True
    if ((x + 1, y) in bombs) and ((x, y) in dead_ends):
        return True

    return False


def in_trap_2(field, bombs, dead_ends, x, y):
    near_tile_1 = (x, y - 1)
    if ((x, y + 1) in bombs) and (field[x - 1, y] != 0) and (field[x + 1, y] != 0) and (near_tile_1 in dead_ends):
        return True

    near_tile_2 = (x + 1, y)
    if ((x - 1, y) in bombs) and (field[x, y - 1] != 0) and (field[x, y + 1] != 0) and (near_tile_2 in dead_ends):
        return True

    near_tile_3 = (x, y + 1)
    if ((x, y - 1) in bombs) and (field[x + 1, y] != 0) and (field[x - 1, y] != 0) and (near_tile_3 in dead_ends):
        return True

    near_tile_4 = (x - 1, y)
    if ((x + 1, y) in bombs) and (field[x, y + 1] != 0) and (field[x, y - 1] != 0) and (near_tile_4 in dead_ends):
        return True

    return False


def in_trap_3(field, bombs, dead_ends, x, y):
    if y-2 >= 0:
        near_tile_1 = (x, y - 2)
        if ((x, y + 1) in bombs) and (field[x - 1, y] != 0) and (field[x + 1, y] != 0)and (field[x - 1, y-1] != 0) and (field[x + 1, y-1] != 0) and (near_tile_1 in dead_ends):
            return True

    if x+2 <= 16:
        near_tile_2 = (x + 2, y)
        if ((x - 1, y) in bombs) and (field[x, y - 1] != 0) and (field[x, y + 1] != 0) and (field[x+1, y - 1] != 0) and (field[x+1, y + 1] != 0) and (near_tile_2 in dead_ends):
            return True

    if y+2 <= 16:
        near_tile_3 = (x, y + 2)
        if ((x, y - 1) in bombs) and (field[x + 1, y] != 0) and (field[x - 1, y] != 0) and (field[x + 1, y+1] != 0) and (field[x - 1, y+1] != 0)and (near_tile_3 in dead_ends):
            return True

    if x-2>=0:
        near_tile_4 = (x - 2, y)
        if ((x + 1, y) in bombs) and (field[x, y + 1] != 0) and (field[x, y - 1] != 0) and (field[x-1, y + 1] != 0) and (field[x-1, y - 1] != 0)and (near_tile_4 in dead_ends):
            return True

    return False



