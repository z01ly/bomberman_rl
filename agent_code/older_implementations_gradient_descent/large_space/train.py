from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, decay_schedule

import numpy as np


def setup_training(self):
    self.gamma = 0.7
    # self.alpha = 0.5
    self.init_alpha = 0.05
    self.min_alpha = 0.01
    self.alpha_decay_ratio = 0.5
    self.alphas = decay_schedule(self, self.init_alpha, self.min_alpha, self.alpha_decay_ratio, self.n_episodes)

    self.action_dict = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}

    # initial gradient
    self.gradient = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    if old_game_state == None:
        return

    reward = reward_from_events(self, events)
    new_state_feature = state_to_features(new_game_state)
    old_state_feature = state_to_features(old_game_state)
    action_int = self.action_dict[self_action]

    max_q = np.max(self.model @ new_state_feature)
    q_func = self.model @ old_state_feature
    q_func = np.squeeze(q_func)
    cur_q = q_func[action_int]

    self.gradient = np.zeros((6,2313))
    self.gradient[action_int,:] = (-1)*old_state_feature/2313
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

    reward = reward_from_events(self, events)
    last_state_feature = state_to_features(last_game_state)
    action_int = self.action_dict[last_action]

    self.gradient = np.zeros((6, 2313))
    self.gradient[action_int, :] = (-1) * last_state_feature / 2313

    round_n = last_game_state['round'] - 1
    self.model = self.model + self.alphas[round_n] * reward * self.gradient


    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.WAITED: 0,
        e.INVALID_ACTION: -1,
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0.11,
        e.CRATE_DESTROYED: 0.5,
        e.COIN_FOUND: 0.1,
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -10,
        e.GOT_KILLED: -8,
        e.OPPONENT_ELIMINATED: 1,
        e.SURVIVED_ROUND: 10,

    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
