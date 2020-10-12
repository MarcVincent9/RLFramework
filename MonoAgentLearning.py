# -*- coding: utf-8 -*-

import random
import numpy as np
from util import timeout, roulette

DEFAULT_TIME_LIMIT = 3600


class MDPLearning:
    """gamma < 1 ensures termination!"""

    DEFAULT_THETA = .001

    def __init__(self, markov_decision_process):
        self.mdp = markov_decision_process
        self.state_values, self.q_values, self.policy = None, None, None
        self.step, self.episode, self.time = None, None, None

    def initialize_state_values(self):
        self.state_values = dict.fromkeys(self.mdp.states(), 0)

    def stepsize(self, timestep, state, action):
        return .1

    # Dynamic programming

    def expectation(self, state, action):
        return sum(p * (r + self.mdp.gamma(state) * self.state_values[s2])
                   for (s2, r), p in self.mdp.expected_transition(state, action).items())

    @timeout
    def policy_evaluation(self, policy, theta=DEFAULT_THETA, time_limit=DEFAULT_TIME_LIMIT):
        self.initialize_state_values()
        delta = theta
        while delta >= theta:
            delta = 0
            for s in self.mdp.states():
                if not self.mdp.is_state_terminal(s):
                    v = self.state_values[s]
                    self.state_values[s] = sum(policy[s][a] * self.expectation(s, a) for a in self.mdp.actions(s))
                    delta = max(delta, abs(v - self.state_values[s]))

    @timeout
    def policy_iteration(self, theta=DEFAULT_THETA, time_limit=DEFAULT_TIME_LIMIT):
        """gamma < 1 ensures termination!"""
        self.initialize_state_values()
        self.policy = {state: random.choice(self.mdp.actions(state)) for state in self.mdp.states()
                       if not self.mdp.is_state_terminal(state)}
        optimum = False
        while not optimum:
            # policy evaluation
            delta = theta
            while delta >= theta:
                delta = 0
                for s in self.mdp.states():
                    if not self.mdp.is_state_terminal(s):
                        v = self.state_values[s]
                        self.state_values[s] = self.expectation(s, self.policy[s])
                        delta = max(delta, abs(v - self.state_values[s]))
            # policy improvement
            optimum = True
            for s in self.mdp.states():
                if not self.mdp.is_state_terminal(s):
                    former_choice = self.policy[s]
                    self.policy[s] = max(self.mdp.actions(s), key=lambda a: self.expectation(s, a))
                    optimum = optimum and self.policy[s] == former_choice

    @timeout
    def value_iteration(self, theta=DEFAULT_THETA, time_limit=DEFAULT_TIME_LIMIT):
        """gamma < 1 ensures termination!"""
        self.initialize_state_values()
        delta = theta
        while delta >= theta:
            delta = 0
            for s in self.mdp.states():
                if not self.mdp.is_state_terminal(s):
                    v = self.state_values[s]
                    self.state_values[s] = max(self.expectation(s, a) for a in self.mdp.actions(s))
                    delta = max(delta, abs(v - self.state_values[s]))

    def policy_improvement(self):
        self.policy = {state: max(self.mdp.actions(state), key=lambda a: self.expectation(state, a))
                       for state in self.mdp.states() if not self.mdp.is_state_terminal(state)}

    # State value learning

    @timeout
    def one_step_td(self, policy, time_limit=DEFAULT_TIME_LIMIT,
                    step_limit=np.infty, episode_limit=np.infty, verbose=False):
        mdp = self.mdp
        self.step, self.episode = 0, 0
        self.initialize_state_values()
        state = mdp.sample_initial_state()
        if verbose:
            print("Initial state:\n" + mdp.to_string(state))
        while self.step < step_limit and self.episode < episode_limit:
            action = roulette(policy[state])
            next_state, reward = mdp.sample_transition(state, action)
            self.step += 1
            if verbose:
                print("Time step (episode): %d (%d)" % (self.step, self.episode))
                print("Action: " + str(action))
                print("Reward: " + str(reward))
                print("New state:\n" + mdp.to_string(next_state))
            if mdp.is_state_terminal(next_state):
                next_state = mdp.sample_initial_state()
                if verbose:
                    print("End of episode. Initial state:\n" + mdp.to_string(next_state))
                self.episode += 1
                gamma = 0
            else:
                gamma = mdp.gamma(next_state)
            stepsize = self.stepsize(self.step, state, action)
            target = reward + gamma * self.state_values[next_state]
            self.state_values[state] += stepsize * (target - self.state_values[state])
            state = next_state

    @timeout
    def n_step_td(self, policy, bootstrapping_depth, time_limit=DEFAULT_TIME_LIMIT,
                  step_limit=np.infty, episode_limit=np.infty, verbose=False):
        mdp = self.mdp
        self.step, self.episode, episode_beginning = 0, 0, 0
        self.initialize_state_values()
        state = mdp.sample_initial_state()
        if verbose:
            print("Initial state:\n" + mdp.to_string(state))
        terminal_flag = False
        update_timestep = episode_beginning - bootstrapping_depth
        last_rewards, last_states, last_stepsizes, last_gammas = [], [], [], [1.]
        while self.step < step_limit and self.episode < episode_limit:
            if not terminal_flag:
                action = roulette(policy[state])
                next_state, reward = mdp.sample_transition(state, action)  # movement
                self.step += 1
                if verbose:
                    print("Time step (episode): %d (%d)" % (self.step, self.episode))
                    print("Action: " + str(action))
                    print("Reward: " + str(reward))
                    print("New state:\n" + mdp.to_string(next_state))
                if mdp.is_state_terminal(next_state):
                    terminal_flag = True
                    next_state = mdp.sample_initial_state()
                    self.episode += 1
                    if verbose:
                        print("End of episode. Initial state:\n" + mdp.to_string(next_state))
                last_gammas.append(mdp.gamma(next_state) * last_gammas[-1])
                last_rewards.append(reward)
                last_states.append(state)
                last_stepsizes.append(self.stepsize(self.step, state, action))
                state = next_state
            update_timestep += 1
            if update_timestep >= episode_beginning:
                np_gammas = np.array(last_gammas)
                expected_return = np.sum(np.array(last_rewards) * np_gammas[:-1])
                if not terminal_flag:
                    expected_return += np_gammas[-1] * self.state_values[next_state]
                last_rewards.pop(0)
                stepsize = last_stepsizes.pop(0)
                updated_state = last_states.pop(0)
                last_gammas = (np_gammas[1:] / np_gammas[1]).tolist()
                last_gammas[0] = 1  # avoid rounding issues
                td_error = expected_return - self.state_values[updated_state]
                self.state_values[updated_state] += stepsize * td_error  # update
            if terminal_flag and update_timestep == self.step - 1:
                episode_beginning = self.step
                terminal_flag = False
                update_timestep = episode_beginning - bootstrapping_depth

    # action value learning

    def initialize_q_values(self):
        self.q_values = {state: dict.fromkeys(self.mdp.actions(state), 0) for state in self.mdp.states()}

    def behavior_policy_action(self, state, **kwargs):
        """
        Return an action selected according to the behavior policy.
        Default: epsilon-greedy.
        """
        epsilon = kwargs["behavior_epsilon"]
        if np.random.rand() < epsilon:
            return random.choice(self.mdp.actions(state))
        return max(self.mdp.actions(state), key=lambda action: self.q_values[state][action])

    def target_policy_expectation(self, state, **kwargs):
        """
        Return the expectation of the Q-value over all possible actions in this state according to the target policy.
        If not implemented, the learning algorithm will be Sarsa; else, Expected Sarsa.
        """
        raise NotImplementedError

    def greedy_sarsa_expectation(self, state, **kwargs):
        """Can be used as target_policy_expectation to implement Expected Sarsa with an epsilon-greedy target policy."""
        epsilon = kwargs["target_epsilon"]
        actions = self.mdp.actions(state)
        greedy_q_value = max(self.q_values[state][action] for action in actions)
        if epsilon == 0:
            return greedy_q_value
        expected_exploratory_q_value = sum(self.q_values[state][action] for action in actions) / len(actions)
        return epsilon * expected_exploratory_q_value + (1 - epsilon) * greedy_q_value

    def q_learning_expectation(self, state, **kwargs):
        """Can be used as target_policy_expectation to implement Q-learning"""
        return self.greedy_sarsa_expectation(state, target_epsilon=0)

    def get_policy(self):
        self.policy = {state: max(self.mdp.actions(state), key=lambda action: self.q_values[state][action])
                       for state in self.mdp.states() if not self.mdp.is_state_terminal(state)}

    @timeout
    def one_step_action_learning(self, time_limit=DEFAULT_TIME_LIMIT, step_limit=np.infty, episode_limit=np.infty,
                                 verbose=False, **kwargs):
        mdp = self.mdp
        self.step, self.episode = 0, 0
        self.initialize_q_values()
        state = mdp.sample_initial_state()
        action = self.behavior_policy_action(state, **kwargs)
        if verbose:
            print("Initial state:\n" + mdp.to_string(state))
        while self.step < step_limit and self.episode < episode_limit:
            next_state, reward = mdp.sample_transition(state, action)
            self.step += 1
            if verbose:
                print("Time step (episode): %d (%d)" % (self.step, self.episode))
                print("Action: " + str(action))
                print("Reward: " + str(reward))
                print("New state:\n" + mdp.to_string(next_state))
            if mdp.is_state_terminal(next_state):
                next_state = mdp.sample_initial_state()
                self.episode += 1
                if verbose:
                    print("End of episode. Initial state:\n" + mdp.to_string(next_state))
                gamma = 0
            else:
                gamma = mdp.gamma(next_state)
            next_action = self.behavior_policy_action(next_state, **kwargs)
            if gamma:
                try:
                    next_q = self.target_policy_expectation(next_state, **kwargs)
                except NotImplementedError:
                    next_q = self.q_values[next_state][next_action]
                target = reward + gamma * next_q
            else:
                target = reward
            stepsize = self.stepsize(self.step, state, action)
            self.q_values[state][action] += stepsize * (target - self.q_values[state][action])
            state = next_state
            action = next_action

    @timeout
    def n_step_action_learning(self, bootstrapping_depth, time_limit=DEFAULT_TIME_LIMIT,
                               step_limit=np.infty, episode_limit=np.infty, verbose=False, **kwargs):
        mdp = self.mdp
        self.step, self.episode, episode_beginning = 0, 0, 0
        self.initialize_q_values()
        state = mdp.sample_initial_state()
        action = self.behavior_policy_action(state, **kwargs)
        if verbose:
            print("Initial state:\n" + mdp.to_string(state))
        terminal_flag = False
        update_timestep = episode_beginning - bootstrapping_depth
        last_rewards, last_states, last_actions, last_stepsizes, last_gammas = [], [], [], [], [1.]
        while self.step < step_limit and self.episode < episode_limit:
            if not terminal_flag:
                next_state, reward = mdp.sample_transition(state, action)
                last_rewards.append(reward)
                last_states.append(state)
                last_actions.append(action)
                last_stepsizes.append(self.stepsize(self.step, state, action))
                self.step += 1
                if verbose:
                    print("Time step (episode): %d (%d)" % (self.step, self.episode))
                    print("Action: " + str(action))
                    print("Reward: " + str(reward))
                    print("New state:\n" + mdp.to_string(next_state))
                if mdp.is_state_terminal(next_state):
                    terminal_flag = True
                    next_state = mdp.sample_initial_state()
                    self.episode += 1
                    if verbose:
                        print("End of episode. Initial state:\n" + mdp.to_string(next_state))
                else:
                    action = self.behavior_policy_action(next_state, **kwargs)
                last_gammas.append(mdp.gamma(next_state) * last_gammas[-1])
                state = next_state
            update_timestep += 1
            if update_timestep >= episode_beginning:
                np_gammas = np.array(last_gammas)
                expected_return = np.sum(np.array(last_rewards) * np_gammas[:-1])
                if not terminal_flag:
                    try:
                        next_q = self.target_policy_expectation(next_state, **kwargs)
                    except NotImplementedError:
                        next_q = self.q_values[next_state][action]
                    expected_return += np_gammas[-1] * next_q
                last_rewards.pop(0)
                stepsize = last_stepsizes.pop(0)
                updated_state = last_states.pop(0)
                updated_action = last_actions.pop(0)
                last_gammas = (np_gammas[1:] / np_gammas[1]).tolist()
                last_gammas[0] = 1  # avoid rounding issues
                td_error = expected_return - self.q_values[updated_state][updated_action]
                self.q_values[updated_state][updated_action] += stepsize * td_error  # update
            if terminal_flag and update_timestep == self.step - 1:
                episode_beginning = self.step
                terminal_flag = False
                update_timestep = episode_beginning - bootstrapping_depth
                action = self.behavior_policy_action(next_state, **kwargs)
