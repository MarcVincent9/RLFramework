# -*- coding: utf-8 -*-

import random
import numpy as np
from util import timeout

DEFAULT_TIME_LIMIT = 3600


class SDGLearning:

    def __init__(self, sequential_markov_game, player_classes=None):
        self.smg = sequential_markov_game
        self.step, self.episode, self.time = None, None, None
        _player_classes = dict.fromkeys(self.smg.agents(), LearningSDGPlayer)
        if player_classes is not None:
            _player_classes.update(player_classes)
        self.players = {player: _player_classes[player](self.smg, i) for i, player in enumerate(self.smg.agents())}

    @timeout
    def one_step_afterstate_learning(self, time_limit=DEFAULT_TIME_LIMIT, step_limit=np.infty, episode_limit=np.infty,
                                     verbose=False, **kwargs):
        smg = self.smg
        self.step, self.episode = 0, 0
        for player in self.players.values():
            player.initialize_afterstate_values()
        state = smg.sample_initial_state()
        if verbose:
            print("Initial state:\n" + smg.to_string(state))
        player = self.players[self.smg.turn(state)]
        action = player.behavior_policy_action(state, **kwargs)
        player.update_afterstate_values(state, action)  # initialize last state and action
        while self.step < step_limit and self.episode < episode_limit:
            state, *rewards = smg.certain_transition(state, action)
            for i, player in enumerate(self.players.values()):
                player.feed_reward(rewards[i])
            self.step += 1
            if verbose:
                print("Time step (episode): %d (%d)" % (self.step, self.episode))
                print("Action: " + str(action))
                print("Rewards: " + str(rewards))
                print("New state:\n" + smg.to_string(state))
            if smg.is_state_terminal(state):
                for player in self.players.values():
                    player.update_afterstate_values()
                state = smg.sample_initial_state()
                self.episode += 1
                if verbose:
                    print("End of episode. Initial state:\n" + smg.to_string(state))
            player = self.players[self.smg.turn(state)]
            action = player.behavior_policy_action(state, **kwargs)
            player.update_afterstate_values(state, action)

    def get_policies(self):
        for player in self.players.values():
            player.get_policy()


class SDGPlayer:

    def __init__(self, sequential_markov_game, agent_number):
        self.smg = sequential_markov_game
        self.num = agent_number

    def behavior_policy_action(self, state, **kwargs):
        raise NotImplementedError

    def initialize_afterstate_values(self):
        pass

    def feed_reward(self, reward):
        pass

    def update_afterstate_values(self, state=None, action=None):
        pass


class HumanSDGPlayer(SDGPlayer):

    def behavior_policy_action(self, state, **kwargs):
        actions = self.smg.actions(state)
        print("Possible actions:")
        for i, action in enumerate(actions):
            print("%d: %s"%(i, action))
        return actions[int(input("Your choice (by number): "))]


class LearningSDGPlayer(SDGPlayer):

    DEFAULT_EPSILON = .1

    def __init__(self, sequential_markov_game, agent_number):
        super().__init__(sequential_markov_game, agent_number)
        self.last_state, self.last_action, self.total_reward = None, None, 0
        self.afterstate_values, self.policy = None, None

    def greedy_action(self, state):
        try:
            return max(self.smg.actions(state),
                       key=lambda action: self.afterstate_values[self.smg.certain_transition(state, action)[0]])
        except KeyError:
            return None

    def behavior_policy_action(self, state, **kwargs):
        epsilon = kwargs.get("behavior_epsilon_%d"%self.num, self.DEFAULT_EPSILON)
        if np.random.rand() < epsilon:
            return random.choice(self.smg.actions(state))
        return self.greedy_action(state)

    def initialize_afterstate_values(self):
        self.afterstate_values = {self.smg.certain_transition(state, action)[0]: 0
                                  for state in self.smg.states() if self.smg.turn(state) == self.smg.agents()[self.num]
                                  and not self.smg.is_state_terminal(state) for action in self.smg.actions(state)}

    def stepsize(self, state, action):
        return .1

    def feed_reward(self, reward):
        self.total_reward += reward

    def update_afterstate_values(self, state=None, action=None):
        """If state and action are None, terminal update"""
        if self.last_state is not None:
            last_afterstate = self.smg.certain_transition(self.last_state, self.last_action)[0]
            stepsize = self.stepsize(self.last_state, self.last_action)
            target = self.total_reward
            if state is not None:
                afterstate = self.smg.certain_transition(state, action)[0]
                target += self.smg.gamma(state) * self.afterstate_values[afterstate]
            self.afterstate_values[last_afterstate] += stepsize * (target - self.afterstate_values[last_afterstate])
        self.last_state, self.last_action, self.total_reward = state, action, 0

    def get_policy(self):
        self.policy = {state: {self.greedy_action(state): 1} for state in self.smg.states()
                       if not self.smg.is_state_terminal(state)}
