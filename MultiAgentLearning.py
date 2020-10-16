# -*- coding: utf-8 -*-

from MarkovTypes import *
from util import timeout
import numpy as np
import random

DEFAULT_TIME_LIMIT = 3600


class ExperienceGenerator:

    def __init__(self, markov_game):
        self.mg = markov_game
        self.step, self.episode, self.time = None, None, None
        self.players = {}

    def add_player(self, player_name, player_class, learning=True, **kwargs):
        if player_name not in self.mg.agents():
            raise ValueError("No player by this name")
        self.players[player_name] = player_class(self.mg, player_name, learning, **kwargs)

    def _update_values_all(self, state, actions):
        for player in self.players.values():
            if player.learning:
                player.update_values(state, actions)

    def _feed_rewards(self, rewards):
        for i, name in enumerate(self.mg.agents()):
            self.players[name].feed_reward(rewards[i])

    def _action_choice_all(self, state):
        return {name: player.behavior_policy(state) if player.learning else player.target_policy(state)
                for name, player in self.players.items()}

    @timeout
    def generate_experience(self, time_limit=DEFAULT_TIME_LIMIT, step_limit=np.infty, episode_limit=np.infty,
                            verbose=False):
        mg = self.mg
        assert set(mg.agents()) == set(self.players.keys())
        print("Agents:")
        for name, player in self.players.items():
            assert name == player.name
            assert mg == player.mg
            print("> %s: %s" % (name, str(player)))
        self.step, self.episode = 0, 0
        for player in self.players.values():
            player.initialize()
        state = mg.sample_initial_state()
        if verbose:
            print("Initial state:\n%s\n" % mg.to_string(state))
        actions = self._action_choice_all(state)
        self._update_values_all(state, actions)  # initialize last state and action
        while self.step < step_limit and self.episode < episode_limit:
            state, *rewards = mg.multi_sample_transition(state, actions)
            self._feed_rewards(rewards)
            self.step += 1
            if verbose:
                print("Time step (episode): %d (%d)" % (self.step, self.episode))
                print("Actions: %s" % actions)
                print("Rewards: %s" % rewards)
                print("New state:\n%s\n" % mg.to_string(state))
            if mg.is_state_terminal(state):
                self._update_values_all(None, None)
                state = mg.sample_initial_state()
                self.episode += 1
                if verbose:
                    print("End of episode. Initial state:\n%s\n" % mg.to_string(state))
            actions = self._action_choice_all(state)
            self._update_values_all(state, actions)


class HumanPlayer:

    def __init__(self, markov_game, agent_name, learning):
        self.mg = markov_game
        self.name = agent_name
        self.learning = False
        self.last_rewards = []

    def __str__(self):
        return "HumanPlayer"

    def target_policy(self, state):
        actions = self.mg.multi_actions(state, self.name)
        if len(actions) == 1:
            return actions[0]
        print("Possible actions:")
        for i, action in enumerate(actions):
            print("%d: %s" % (i, action))
        choice = actions[int(input("Your choice (by number): "))]
        print()
        return choice

    def initialize(self):
        self.last_rewards = []

    def feed_reward(self, reward):
        self.last_rewards.append(reward)


class ArtificialPlayer:

    def __init__(self, markov_game, agent_name, learning):
        self.mg = markov_game
        self.name = agent_name
        self.learning = learning
        # self.last_states, self.last_actions, self.last_rewards = [], [], []

    def __str__(self):
        return "%s: learning=%s" % (type(self).__name__, str(self.learning))

    def target_policy(self, state):
        return random.choice(self.mg.multi_actions(state, self.name))

    def behavior_policy(self, state):
        raise NotImplementedError

    def initialize(self):
        pass

    def feed_reward(self, reward):
        pass

    def _stepsize(self, state, action):
        return .1

    def update_values(self, state, actions):
        """If state and action are None, terminal update."""
        pass


class SDGTabularArtificialPlayer(ArtificialPlayer):
    """Implements tabular afterstate one-step Sarsa"""

    def __init__(self, sequential_deterministic_game, agent_name, learning, epsilon=.1):
        assert isinstance(sequential_deterministic_game, SequentialDeterministicGame)
        super().__init__(sequential_deterministic_game, agent_name, learning)
        self.last_state, self.last_action, self.total_reward = None, None, 0
        self.afterstate_values = None
        self.epsilon = epsilon

    def __str__(self):
        return  "%s, epsilon=%f" % (super().__str__(), self.epsilon)

    def target_policy(self, state):
        try:
            return max(self.mg.actions(state),
                       key=lambda action: self.afterstate_values[self.mg.certain_transition(state, action)[0]])
        except KeyError:
            return None

    def behavior_policy(self, state):
        if self.mg.turn(state) != self.name:
            return None
        if np.random.rand() < self.epsilon:
            return random.choice(self.mg.actions(state))
        return self.target_policy(state)

    def initialize(self):
        self.last_state, self.last_action, self.total_reward = None, None, 0
        if self.learning:
            self.afterstate_values = {self.mg.certain_transition(state, action)[0]: 0
                                      for state in self.mg.states() if self.mg.turn(state) == self.name
                                      and not self.mg.is_state_terminal(state) for action in self.mg.actions(state)}

    def feed_reward(self, reward):
        self.total_reward += reward

    def update_values(self, state, actions):
        # nothing to do if it's not the player's turn or not the end of an episode
        if state is None or self.mg.turn(state) == self.name:
            action = actions[self.name] if actions is not None else None
            if self.last_state is not None:  # update the previous state-action pair value
                last_afterstate = self.mg.certain_transition(self.last_state, self.last_action)[0]
                stepsize = self._stepsize(self.last_state, self.last_action)
                target = self.total_reward
                if state is not None:  # no bootstrapping at the end of an episode
                    afterstate = self.mg.certain_transition(state, action)[0]
                    target += self.mg.gamma(state) * self.afterstate_values[afterstate]
                self.afterstate_values[last_afterstate] += stepsize * (target - self.afterstate_values[last_afterstate])
            self.last_state, self.last_action, self.total_reward = state, action, 0
