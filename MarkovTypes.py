# -*- coding: utf-8 -*-

import itertools as it
import numpy as np
from util import roulette

LONE_STATE = "state"
LONE_AGENT = "agent"


class MarkovGame:
    """Generic class for Markov games.
    Mandatory implementation: (states,) sample_initial_state, is_state_terminal, agents, multi_actions,
    multi_expected_transition or multi_sample_transition
    """

    def agents(self):
        """Return the list of all agents' ID.

        :rtype: list of integers/strings
        """
        raise NotImplementedError

    def nb_agents(self):
        """Return the number of agents.

        :rtype: integer
        """
        return len(self.agents())

    def sample_initial_state(self):
        """Return an initial state (according to the probability distribution over initial states).

        :rtype: state
        """
        raise NotImplementedError

    def is_state_terminal(self, state):
        """Return whether this state is terminal.

        :rtype: bool
        """
        return False

    def multi_actions(self, state, agent):
        """Return the list of all the agent's possible actions in given state.

        :param state
        :param agent
        :rtype: dictionary {agent: list of (tuples of) integers/strings}
        """
        raise NotImplementedError

    def multi_expected_transition(self, state, actions):
        # TODO pourquoi pas un dico de rewards?
        # multiobjective!
        """Return a probability distribution for the next state and rewards.

        :param state: current state
        :param actions: dictionary (key: agent; value: action)
        :rtype: dictionary {tuple(state, *rewards): probability} # rewards listed in the same order as the agents
        """
        raise NotImplementedError

    def multi_sample_transition(self, state, actions):
        """Return the next state and rewards according to the transition distribution.

        :param state: current state
        :param actions: dictionary (key: agent; value: action)
        :rtype: tuple(state, *rewards) # rewards listed in the same order as the list of agents
        """
        return roulette(self.multi_expected_transition(state, actions))

    def gamma(self, state):
        """Return the discount factor.

        :rtype: float in [0, 1]
        """
        return 1

    def states(self):
        """Return the list of all states.

        :rtype: list of (tuples of) integers/strings
        """
        raise NotImplementedError

    def check_transitions(self, epsilon):
        """
        verify that all transitions are valid probability distributions.

        :param epsilon: error rate
        """
        for state in self.states():
            agents = self.agents()
            for actions in it.product(self.multi_actions(state, agent) for agent in agents):
                common_action = {agent: action for agent, action in zip(agents, actions)}
                if abs(1 - sum(self.multi_expected_transition(state, common_action).values())) > epsilon:
                    return False
        return True

    # TODO inutile ac MultiAgentLearning ?
    def test_policies(self, policies, step_limit=np.infty, episode_limit=np.infty, verbose=False):
        """Stochastic policies defined for all states!"""
        state = self.sample_initial_state()
        if verbose:
            print("Initial state:\n" + self.to_string(state))
        list_returns = [np.zeros(self.nb_agents())]
        timestep, episode = 0, 0
        while timestep < step_limit:
            actions = {agent: roulette(policies[agent][state]) for agent in self.agents()}
            state, rewards = self.multi_sample_transition(state, actions)
            list_returns[-1] += np.array(rewards)
            timestep += 1
            if verbose:
                print("Time step (episode): %d (%d)" % (timestep, episode))
                print("Action: " + str(actions))
                print("Reward: " + str(rewards))
                print("Total return: " + str(list_returns[-1]))
                print("New state:\n" + self.to_string(state))
            if self.is_state_terminal(state):
                state = self.sample_initial_state()
                if verbose:
                    print("End of episode. Initial state:\n" + self.to_string(state))
                list_returns.append(np.zeros(self.nb_agents()))
                episode += 1
                if episode == episode_limit:
                    break
        return np.array(list_returns)

    def to_string(self, state):
        return str(state)


class DeterministicGame(MarkovGame):
    """Generic class for deterministic games.
    Mandatory implementation: (states,) sample_initial_state, is_state_terminal, agents, multi_actions,
    multi_certain_transition or multi_sample_transition
    """

    def multi_certain_transition(self, state, actions):
        """Return the next state and rewards.

        :param state: current state
        :param actions: dictionary (key: agent; value: action)
        :rtype: tuple(state, *rewards) # rewards listed in the same order as the agents
        """
        raise NotImplementedError

    def multi_expected_transition(self, state, actions):
        return {self.multi_certain_transition(state, actions): 1}

    def multi_sample_transition(self, state, actions):
        return self.multi_certain_transition(state, actions)


class SequentialMarkovGame(MarkovGame):
    """Generic class for sequential Markov games.
    Mandatory implementation: (states,) sample_initial_state, is_state_terminal, agents, actions,
    expected_transition or sample_transition, turn
    """

    def turn(self, state):
        """Return the ID of the agent whose turn it is to play in the current state.

        :param state: state
        :rtype: integer/string
        """
        raise NotImplementedError

    def actions(self, state):
        """Return the list of the player's possible actions in given state.

        :param state: state
        :rtype: list of (tuples of) integers/strings
        """
        raise NotImplementedError

    def multi_actions(self, state, agent):
        if agent == self.turn(state):
            return self.actions(state)
        return [None]

    def expected_transition(self, state, action):
        """Return a probability distribution for the next state and reward
        according to the current state and player's action.

        :param state: current state
        :param action: action chosen by the player
        :rtype: dictionary {(state, *rewards): probability}
        """
        raise NotImplementedError

    def multi_expected_transition(self, state, actions):
        return self.expected_transition(state, actions[self.turn(state)])

    def sample_transition(self, state, action):
        """Return the next state and reward according to the transition distribution
        and to the current state and player's action.

        :param state: current state
        :param action: action chosen by the player
        :rtype: (state, *rewards)
        """
        return roulette(self.expected_transition(state, action))

    def multi_sample_transition(self, state, actions):
        return self.sample_transition(state, actions[self.turn(state)])

    def test_policies(self, policies, step_limit=np.infty, episode_limit=np.infty, verbose=False):
        """Stochastic policies defined for all states where the agent acts!"""
        return super().test_policies(
            {agent: {**dict.fromkeys(self.states()), **policy} for agent, policy in policies.items()},
            step_limit, episode_limit, verbose)


class SequentialDeterministicGame(DeterministicGame, SequentialMarkovGame):
    """Generic class for sequential deterministic games.
    Mandatory implementation: (states,) sample_initial_state, is_state_terminal, agents, actions, certain_transition, turn
    """

    def certain_transition(self, state, action):
        """Return the next state and rewards.

        :param state: current state
        :param action
        :rtype: tuple(state, *rewards) # rewards listed in the same order as the agents
        """
        raise NotImplementedError

    def multi_certain_transition(self, state, actions):
        return self.certain_transition(state, actions[self.turn(state)])

    def expected_transition(self, state, action):
        return {self.certain_transition(state, action): 1}


class MarkovDecisionProcess(SequentialMarkovGame):
    """Generic class for Markov decision processes.
    Mandatory implementation: (states,) sample_initial_state, is_state_terminal, actions,
    expected_transition or sample_transition
    """
    
    def agents(self):
        return [LONE_AGENT]

    def turn(self, state):
        return LONE_AGENT

    def test_policy(self, policy, step_limit=np.infty, episode_limit=np.infty, verbose=False):
        return super(SequentialMarkovGame, self).test_policies({LONE_AGENT: policy}, step_limit, episode_limit, verbose)


class DeterministicDecisionProcess(SequentialDeterministicGame, MarkovDecisionProcess):
    """Generic class for deterministic decision processes.
    Mandatory implementation: (states,) sample_initial_state, is_state_terminal, actions, certain_transition
    """
    pass


class MarkovRewardProcess(MarkovDecisionProcess):
    """Generic class for Markov reward processes.
    Mandatory implementation: (states,) sample_initial_state, is_state_terminal, _expected_transition or _sample_transition
    """

    def actions(self, state):
        return [None]

    def _expected_transition(self, state):
        """Return a probability distribution for the next state and reward.

        :param state: current state
        :rtype: dictionary {(state, reward): probability}
        """
        raise NotImplementedError

    def expected_transition(self, state, action):
        return self._expected_transition(state)

    def _sample_transition(self, state):
        """Return the next state and reward according to the transition distribution.

        :param state: current state
        :rtype: (state, reward)
        """
        return roulette(self._expected_transition(state))

    def sample_transition(self, state, action):
        return self._sample_transition(state)

    def test(self, step_limit=np.infty, episode_limit=np.infty, verbose=False):
        return super().test_policy({LONE_AGENT: dict.fromkeys(self.states())}, step_limit, episode_limit, verbose)


class KArmedBandit(MarkovDecisionProcess):
    """Generic class for k-armed bandits.
    Mandatory implementation: _actions, _sample_transition
    """

    def states(self):
        return [LONE_STATE]

    def sample_initial_state(self):
        return LONE_STATE

    def _actions(self):
        """Return the list of the agent's possible actions.

        :rtype: list of (tuples of) integers/strings
        """
        raise NotImplementedError

    def actions(self, state):
        return self._actions()

    def _sample_transition(self, action):
        """Return the next reward.

        :param action: chosen action
        :rtype: reward
        """
        raise NotImplementedError

    def sample_transition(self, state, action):
        return LONE_STATE, self._sample_transition(action)
