from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, decay_schedule, valid_actions

import numpy as np


def setup_training(self):
    self.gamma = 0.9
    # self.alpha = 0.5
    self.init_alpha = 0.2
    self.min_alpha = 0.05
    self.alpha_decay_ratio = 0.5
    self.alphas = decay_schedule(self, self.init_alpha, self.min_alpha, self.alpha_decay_ratio, self.n_episodes)

    self.action_dict = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}

    # initial gradient
    self.gradient = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    if old_game_state == None:
        return

    reward = reward_from_events(self, events,old_game_state)
    new_state_feature = state_to_features(self.cur_action,new_game_state)
    old_state_feature = state_to_features(self.last_action, old_game_state)
    action_int = self.action_dict[self_action]

    new_q = self.model @ new_state_feature
    new_valid_ac = valid_actions(new_game_state,self)
    invalid_ac = list(set(range(6)) - set(new_valid_ac))
    new_q[invalid_ac] = -100

    max_q = np.max(new_q)
    q_func = self.model @ old_state_feature
    q_func = np.squeeze(q_func)
    cur_q = q_func[action_int]

    self.gradient = np.zeros((6,86))
    self.gradient[action_int,:] = np.sign(old_state_feature)/86
    # self.model = self.model + self.alpha * (reward + self.gamma * max_q - cur_q) * self.gradient
    round_n = old_game_state['round'] - 1
    self.model = self.model + self.alphas[round_n] * (reward + self.gamma * max_q - cur_q) * self.gradient

    return


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    reward = reward_from_events(self, events,last_game_state)
    last_state_feature = state_to_features(self.last_action, last_game_state)
    action_int = self.action_dict[last_action]

    self.gradient = np.zeros((6, 86))
    self.gradient[action_int, :] = np.sign(last_state_feature) / 86

    round_n = last_game_state['round'] - 1
    self.model = self.model + self.alphas[round_n] * reward * self.gradient




    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)
    with open("classic_opp_2.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str],game_state) -> int:
    game_rewards = {
        e.MOVED_LEFT: 0.00,
        e.MOVED_RIGHT: 0.00,
        e.MOVED_UP: 0.00,
        e.MOVED_DOWN: 0.00,
        e.WAITED: 0,
        e.INVALID_ACTION: -0.5,
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0.1,
        e.CRATE_DESTROYED: 0.1,
        e.COIN_FOUND: 1,
        e.COIN_COLLECTED: 3,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -30,
        e.GOT_KILLED: -3,
        e.OPPONENT_ELIMINATED: 0.1,
        e.SURVIVED_ROUND: 0.5,

    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    if self.loop:
        reward_sum -= 1
    if self.trap:
        reward_sum -= 3
    if backwards(self) and not game_state['self'][2] :
        reward_sum -= 0.4


    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def backwards(self):
    if self.last_action == 0 and self.cur_action == 2: return True
    if self.last_action == 2 and self.cur_action == 0: return True
    if self.last_action == 1 and self.cur_action == 3: return True
    if self.last_action == 3 and self.cur_action == 1: return True
    return False
