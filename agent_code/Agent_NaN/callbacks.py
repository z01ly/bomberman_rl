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
        self.model = np.zeros(8)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

    self.init_epsilon = 1.0
    self.min_epsilon = 0.1
    self.epsilon_decay_ratio = 0.9
    self.n_episodes = 3000
    self.epsilons = decay_schedule(self, self.init_epsilon, self.min_epsilon, self.epsilon_decay_ratio, self.n_episodes)

    print(self.model)


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
    self.logger.debug(cur_state)

    # print(qvals)

    qvals = np.squeeze(qvals)
    invalid_ac = list(set(range(6)) - set(val_ac))
    qvals[invalid_ac] = - np.inf
    if game_state['step'] == 1:
        qvals[5] = - np.inf
    a_list = np.argwhere(qvals == np.max(qvals)).flatten().tolist()
    if len(a_list) == 1:
        a = a_list[0]
    else:
        a = random.choice(a_list)

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

    f0 = np.ones(6)
    f1 = into_bomb_range(game_state)
    f2 = away_from_bomb(game_state)
    f3 = cannot_escape(game_state)
    f4 = towards_coin(game_state)
    f5 = away_from_crate(game_state)
    f6 = bomb_crate(game_state)
    # f7 = towards_opp(game_state)
    f8 = bomb_opp(game_state)
    # f9 = corner(game_state)

    features = np.vstack((f0, f1, f2, f3, f4, f5, f6, f8))

    return features


def new_pos(x, y, action):
    if action == 'UP':
        return x, y - 1
    elif action == 'RIGHT':
        return x + 1, y
    elif action == 'DOWN':
        return x, y + 1
    elif action == 'LEFT':
        return x - 1, y
    else:
        # 'WAIT' or 'BOMB'
        return x, y


def bomb_full_range(game_state, x, y):
    field = game_state['field']
    full_range = []
    full_range.append((x, y))

    for i in range(1, 4):
        if field[x - i, y] == -1:
            break
        else:
            full_range.append((x - i, y))

    for i in range(1, 4):
        if field[x + i, y] == -1:
            break
        else:
            full_range.append((x + i, y))

    for i in range(1, 4):
        if field[x, y - i] == -1:
            break
        else:
            full_range.append((x, y - i))

    for i in range(1, 4):
        if field[x, y + i] == -1:
            break
        else:
            full_range.append((x, y + i))

    return full_range


# from fule_based callbacks
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


# f1(s, a) has value 1 if action a will take the agent into the bomb range and has value 0 otherwise
def into_bomb_range(game_state):
    x, y = game_state['self'][3]
    bombs = [i[0] for i in game_state['bombs']]
    vec = np.zeros(6)
    if len(bombs) == 0:
        return vec

    # bomb range
    bomb_range = []
    for (bomb_x, bomb_y) in bombs:
        this_range = bomb_full_range(game_state, bomb_x, bomb_y)
        bomb_range = bomb_range + this_range

    for action in ACTIONS:
        action_int = ACTION_DICT[action]
        if action == 'BOMB':
            vec[action_int] = 1
        else:
            new_x, new_y = new_pos(x, y, action)
            if (new_x, new_y) in bomb_range:
                vec[action_int] = 1

    return vec


# f2(s, a) has value 1 if action a will take the agent away from the nearest bomb and has value 0 otherwise
def away_from_bomb(game_state):
    x, y = game_state['self'][3]
    bombs = [i[0] for i in game_state['bombs']]
    vec = np.zeros(6)
    if len(bombs) == 0:
        return vec

    pos_diff_bombs = [tuple(map(sum, zip(bomb, (-x, -y)))) for bomb in bombs]
    bomb_dist = [sum([abs(ele) for ele in sub]) for sub in pos_diff_bombs]
    nearest_bomb_index = bomb_dist.index(min(bomb_dist))
    nearest_bomb = bombs[nearest_bomb_index]
    dist = np.sum(np.abs(np.array(nearest_bomb) - np.array((x, y))))

    nearest_bomb_range = bomb_full_range(game_state, nearest_bomb[0], nearest_bomb[1])
    if (dist < 4) and ((x, y) in nearest_bomb_range):
        for action in ACTIONS:
            action_int = ACTION_DICT[action]
            if (action != 'BOMB') and (action != 'WAIT'):
                new_x, new_y = new_pos(x, y, action)
                new_dist = np.sum(np.abs(np.array(nearest_bomb) - np.array((new_x, new_y))))
                if new_dist > dist:
                    vec[action_int] = 1

    return vec


# f3(s, a) has value 1 if the agent drop a bomb leading itself to dead end and has value 0 otherwise
def cannot_escape(game_state):
    x, y = game_state['self'][3]
    field = game_state['field']
    bombs = [i[0] for i in game_state['bombs']]
    vec = np.zeros(6)
    if len(bombs) == 0:
        return vec

    dead_ends = trap_area(field)
    for action in ACTIONS:
        action_int = ACTION_DICT[action]
        if action != 'BOMB':
            new_x, new_y = new_pos(x, y, action)
            trap1 = in_trap_1(field, bombs, dead_ends, new_x, new_y)
            trap2 = in_trap_2(field, bombs, dead_ends, new_x, new_y)
            trap3 = in_trap_3(field, bombs, dead_ends, new_x, new_y)
            if (trap1 == True) or (trap2 == True) or (trap3 == True):
                vec[action_int] = 1

    return vec


# f4(s, a) has value 1 if action a will take the agent towards the nearest coin and has value 0 otherwise
def towards_coin(game_state):
    x, y = game_state['self'][3]
    coins = game_state['coins']
    vec = np.zeros(6)
    if len(coins) == 0:
        return vec

    pos_diff_coin = [tuple(map(sum, zip(coin, (-x, -y)))) for coin in coins]
    coin_dist = [sum([abs(ele) for ele in sub]) for sub in pos_diff_coin]
    nearest_coin_index = coin_dist.index(min(coin_dist))
    nearest_coin = np.array(coins[nearest_coin_index])
    dist = np.sum(np.abs(nearest_coin - np.array((x, y))))

    for action in ACTIONS:
        action_int = ACTION_DICT[action]
        if (action != 'BOMB') and (action != 'WAIT'):
            new_x, new_y = new_pos(x, y, action)
            new_dist = np.sum(np.abs(nearest_coin - np.array((new_x, new_y))))
            if new_dist < dist:
                vec[action_int] = 1

    return vec


# f5(s, a) has value 1 if action a will take the agent towards the nearest crate and has value 0 otherwise
def away_from_crate(game_state):
    x, y = game_state['self'][3]
    pos = np.where(game_state['field'] == 1)
    crates = list(zip(pos[0], pos[1]))
    vec = np.zeros(6)
    if len(crates) == 0:
        return vec

    pos_diff_crates = [tuple(map(sum, zip(cur, (-x,-y)))) for cur in crates]
    crate_dist = [sum([abs(ele) for ele in sub]) for sub in pos_diff_crates]
    nearest_crate_index = crate_dist.index(min(crate_dist))
    nearest_crate = np.array(crates[nearest_crate_index])
    dist = np.sum(np.abs(nearest_crate - np.array((x, y))))

    for action in ACTIONS:
        action_int = ACTION_DICT[action]
        if (action != 'BOMB') and (action != 'WAIT'):
            new_x, new_y = new_pos(x, y, action)
            new_dist = np.sum(np.abs(nearest_crate - np.array((new_x, new_y))))
            if new_dist > dist:
                vec[action_int] = 1

    return vec


# f6(s, a) has value 1 if action a will make the agent to bomb the nearest crate and has value 0 otherwise
def bomb_crate(game_state):
    x, y = game_state['self'][3]
    pos = np.where(game_state['field'] == 1)
    crates = list(zip(pos[0], pos[1]))
    vec = np.zeros(6)
    if len(crates) == 0:
        return vec

    pos_diff_crates = [tuple(map(sum, zip(cur, (-x,-y)))) for cur in crates]
    crate_dist = [sum([abs(ele) for ele in sub]) for sub in pos_diff_crates]
    nearest_crate_index = crate_dist.index(min(crate_dist))
    nearest_crate = crates[nearest_crate_index]
    dist = np.sum(np.abs(np.array(nearest_crate) - np.array((x, y))))

    action_int = ACTION_DICT['BOMB']
    if dist < 2:
        vec[action_int] = 1

    return vec


# f7(s, a) has value 1 if action a will take the agent the nearest opponent and has value 0 otherwise
def towards_opp(game_state):
    x, y = game_state['self'][3]
    opp_pos = [i[3] for i in game_state['others']]
    vec = np.zeros(6)
    if len(opp_pos) == 0:
        return vec

    pos_diff_opp = [tuple(map(sum, zip(cur, (-x, -y)))) for cur in opp_pos]
    opp_dist = [sum([abs(ele) for ele in sub]) for sub in pos_diff_opp]
    nearest_opp_index = opp_dist.index(min(opp_dist))
    nearest_opp = np.array(opp_pos[nearest_opp_index])
    dist = np.sum(np.abs(nearest_opp - np.array((x, y))))

    if dist < 6:
        for action in ACTIONS:
            action_int = ACTION_DICT[action]
            if (action != 'BOMB') and (action != 'WAIT'):
                new_x, new_y = new_pos(x, y, action)
                new_dist = np.sum(np.abs(nearest_opp - np.array((new_x, new_y))))
                if new_dist < dist:
                    vec[action_int] = 1

    return vec


# f8(s, a) has value 1 if action a will take the agent to bomb the nearest opponent and has value 0 otherwise
def bomb_opp(game_state):
    x, y = game_state['self'][3]
    opp_pos = [i[3] for i in game_state['others']]
    vec = np.zeros(6)
    if len(opp_pos) == 0:
        return vec

    pos_diff_opp = [tuple(map(sum, zip(cur, (-x, -y)))) for cur in opp_pos]
    opp_dist = [sum([abs(ele) for ele in sub]) for sub in pos_diff_opp]
    nearest_opp_index = opp_dist.index(min(opp_dist))
    nearest_opp = opp_pos[nearest_opp_index]
    dist = np.sum(np.abs(np.array(nearest_opp) - np.array((x, y))))
    # our_bomb_range = bomb_full_range(game_state, x, y)

    action_int = ACTION_DICT['BOMB']
    if dist < 2:
        vec[action_int] = 1
    # if nearest_opp in our_bomb_range:
    #     vec[action_int] = 1

    return vec


def corner(game_state):
    x, y = game_state['self'][3]
    field = game_state['field']
    vec = np.zeros(6)

    action_int = ACTION_DICT['BOMB']
    if ((field[x + 1, y] == 0) and (field[x - 1, y] == -1) and (field[x, y + 1] == 0) and (field[x, y - 1] == -1)) or \
        ((field[x + 1, y] == -1) and (field[x - 1, y] == 0) and (field[x, y + 1] == 0) and (field[x, y - 1] == -1)) or \
        ((field[x + 1, y] == -1) and (field[x - 1, y] == 0) and (field[x, y + 1] == -1) and (field[x, y - 1] == 0)) or \
        ((field[x + 1, y] == 0) and (field[x - 1, y] == -1) and (field[x, y + 1] == -1) and (field[x, y - 1] == 0)):
        vec[action_int] = 1

    return vec
