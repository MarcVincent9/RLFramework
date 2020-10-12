# -*- coding: utf-8 -*-

from MarkovTypes import *
import random
import itertools as it
import numpy as np
import collections


class NormalKArmedBandit(KArmedBandit):

    def __init__(self, n=10):
        self.means = np.random.randn(n)
        self.n = n

    def _actions(self):
        return list(range(self.n))

    def _sample_transition(self, action):
        return np.random.randn() + self.means[action]


class RandomWalk(MarkovRewardProcess):

    def __init__(self, nb_states, leftmost_reward=-1):
        self.nb_states = nb_states + 2
        self.leftmost_reward = leftmost_reward

    def states(self):
        return list(range(self.nb_states))

    def sample_initial_state(self):
        return self.nb_states // 2

    def is_state_terminal(self, state):
        return state == 0 or state == self.nb_states - 1

    def _expected_transition(self, state):
        left_reward = self.leftmost_reward if state - 1 == 0 else 0
        right_reward = 1 if state + 1 == self.nb_states - 1 else 0
        return {(state - 1, left_reward): .5, (state + 1, right_reward): .5}


class GridWorld(DeterministicDecisionProcess):

    def __init__(self, nb_lines, nb_columns, initial=None, win=None, lose=None,  walls=None):
        self.nb_lines = nb_lines
        self.nb_columns = nb_columns
        self.initial = [(0, 0)] if initial is None else initial
        self.win = [(self.nb_lines - 1, self.nb_columns - 1)] if win is None else win
        self.lose = [] if lose is None else lose
        self.walls = [] if walls is None else walls
        self.symbols = {**{state: "_" for state in self.states()},
                        **{state: "I" for state in self.initial},
                        **{state: "W" for state in self.win},
                        **{state: "L" for state in self.lose}}

    def states(self):
        return [s for s in it.product(list(range(self.nb_lines)), list(range(self.nb_columns))) if s not in self.walls]

    def sample_initial_state(self):
        return random.choice(self.initial)

    def is_state_terminal(self, state):
        return state in self.lose or state in self.win

    def actions(self, state):
        return [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def certain_transition(self, state, action):
        next_state = (state[0] + action[0], state[1] + action[1])
        if next_state in self.walls or not(0 <= next_state[0] < self.nb_lines and 0 <= next_state[1] < self.nb_columns):
            next_state = state
        if next_state in self.win:
            reward = 0
        elif next_state in self.lose:
            reward = -10
        else:
            reward = -1
        return next_state, reward

    def gamma(self, state):
        return .999

    def to_string(self, state):
        res = ""
        for l in range(self.nb_lines):
            for c in range(self.nb_columns):
                res += "%s "%("X" if (l, c) == state else self.symbols[l, c])
            res += '\n'
        return res

    def print_policy(self, policy):
        symbols = {(0, 1): '>', (0, -1): '<', (1, 0): 'v', (-1, 0): '^'}
        res = ""
        for l in range(self.nb_lines):
            for c in range(self.nb_columns):
                try:
                    res += "%s " % symbols[policy[l, c]]
                except KeyError:
                    res += "X "
            res += '\n'
        print(res)


class TwoPlayerBoardGame(SequentialDeterministicGame):

    State = collections.namedtuple("State", ["board", "player"])

    def __init__(self):
        self.win, self.draw, self.state_set = set(), set(), {self.sample_initial_state()}
        self.tabular = False

    def states(self):
        if not self.tabular:
            self._subsequent_states(self.sample_initial_state())
            self.tabular = True
        return list(self.state_set)

    def is_state_terminal(self, state):
        """assumes the state was visited by _next_situation!"""
        return state in self.win or state in self.draw

    def agents(self):
        return [0, 1]

    def certain_transition(self, state, action):
        next_state = self._next_situation(state, action)
        if next_state in self.draw:
            rewards = [.5, .5]
        else:
            rewards = [0, 0]
            if next_state in self.win:
                rewards[state.player] = 1
                rewards[next_state.player] = -1
        return (next_state, *rewards)

    def turn(self, state):
        return state.player

    def _subsequent_states(self, state):
        for action in self.actions(state):
            next_state = self._next_situation(state, action)
            if next_state not in self.state_set:
                self.state_set.add(next_state)
                if not self.is_state_terminal(next_state):
                    self._subsequent_states(next_state)

    def _next_situation(self, state, action):
        next_state, changed_position = self._next_state(state, action)
        if not self.tabular:
            if not self.is_state_terminal(next_state):
                if self._is_move_win(next_state, changed_position):
                    self.win.add(next_state)
                elif self._is_state_draw(next_state):
                    self.draw.add(next_state)
        return next_state

    def sample_initial_state(self):
        raise NotImplementedError

    def actions(self, state):
        raise NotImplementedError

    def _next_state(self, state, action):
        """returns the next state and the position on the board that was last altered"""
        raise NotImplementedError

    def _is_move_win(self, next_state, changed_position):
        raise NotImplementedError

    def _is_state_draw(self, state):
        """assumes state is not a win..."""
        raise NotImplementedError


class TicTacToe(TwoPlayerBoardGame):

    EMPTY = '_'
    PLAYERS = 'XO'

    def __init__(self, n=3, d=3):
        self.n = n
        self.d = d
        super().__init__()

    def sample_initial_state(self):
        return self.State(self.EMPTY * self.n ** 2, 0)

    def actions(self, state):
        return [i for i, c in enumerate(state.board) if c == self.EMPTY]

    def to_string(self, state):
        board, player = state
        res = ""
        for i in range(self.n):
            res += ' '.join(board[i * self.n: (i+1) * self.n]) + '\n'
        return res + '(%s)'%self.PLAYERS[player]

    def _next_state(self, state, action):
        next_board = list(state.board)
        next_board[action] = self.PLAYERS[state.player]
        return self.State(''.join(next_board), (state.player + 1) % 2), (action // self.n, action % self.n)

    def _is_move_win(self, next_state, changed_position):
        cp = changed_position
        target = self.PLAYERS[(next_state.player + 1) % 2] * self.d
        npboard = np.resize(list(next_state.board), (self.n, self.n))
        for line in [npboard[cp[0], max(0, cp[1] - self.d + 1): cp[1] + self.d],
                     npboard[max(0, cp[0] - self.d + 1): cp[0] + self.d, cp[1]],
                     np.diag(npboard, k=cp[1] - cp[0])[max(0, min(cp) - self.d + 1): min(cp) + self.d],
                     np.diag(np.fliplr(npboard), k=self.n - 1 - cp[1] - cp[0])[
                     max(0, min(cp[0], self.n - 1 - cp[1]) - self.d + 1): min(cp) + self.d]]:
            if target in ''.join(line):
                return True
        return False

    def _is_state_draw(self, state):
        return state.board.find(self.EMPTY) == -1


class ConnectN(TwoPlayerBoardGame):

    EMPTY = '_'
    PLAYERS = 'RW'

    def __init__(self, width=7, height=6, n=4):
        self.n = n
        self.width = width
        self.height = height
        super().__init__()

    def sample_initial_state(self):
        return self.State((self.EMPTY * self.height,) * self.width, 0)

    def actions(self, state):
        return [i for i, column in enumerate(reversed(state.board)) if column.find(self.EMPTY) >= 0]

    def to_string(self, state):
        res = ""
        for h in reversed(range(self.height)):
            for column in reversed(state.board):
                res += column[h] + ' '
            res += '\n'
        return res + "(%s)" % state.player

    def _next_state(self, state, action):
        i_column = self.width - 1 - action
        next_board = list(state.board)
        top = next_board[i_column].find(self.EMPTY)
        next_board[i_column] = list(next_board[i_column])
        next_board[i_column][top] = self.PLAYERS[state.player]
        next_board[i_column] = ''.join(next_board[i_column])
        return self.State(tuple(next_board), (state.player + 1) % 2), (top, i_column)

    def _is_move_win(self, next_state, changed_position):
        cp = changed_position
        target = self.PLAYERS[(next_state.player + 1) % 2] * self.n
        npboard = np.resize(list(''.join(next_state.board)), (self.width, self.height)).T
        for line in [npboard[cp[0], max(0, cp[1] - self.n + 1): cp[1] + self.n],
                     npboard[max(0, cp[0] - self.n + 1): cp[0] + 1, cp[1]],
                     np.diag(npboard, k=cp[1] - cp[0])[max(0, min(cp) - self.n + 1): min(cp) + self.n],
                     np.diag(np.fliplr(npboard), k=self.width - 1 - cp[1] - cp[0])[
                     max(0, min(cp[0], self.width - 1 - cp[1]) - self.n + 1): min(cp) + self.n]]:
            if target in ''.join(line):
                return True
        return False

    def _is_state_draw(self, state):
        return ''.join(state.board).find(self.EMPTY) == -1


class Go(TwoPlayerBoardGame):

    EMPTY = '_'
    PLAYERS = 'BW'

    def __init__(self, n=19):
        self.n = n
        super().__init__()

    def sample_initial_state(self):
        return self.State(self.EMPTY * self.n ** 2, 0)

    def actions(self, state):
        return [i for i, c in enumerate(state.board) if c == self.EMPTY]  # + abandon

    def to_string(self, state):
        board, player = state
        res = ""
        for i in range(self.n):
            res += ' '.join(board[i * self.n: (i+1) * self.n]) + '\n'
        return res + '(%s)'%self.PLAYERS[player]

    def _next_state(self, state, action):
        next_board = list(state.board)
        next_board[action] = self.PLAYERS[state.player]
        return self.State(''.join(next_board), (state.player + 1) % 2), (action // self.n, action % self.n)

    def _is_move_win(self, next_state, changed_position):
        cp = changed_position
        # TODO

    def _is_state_draw(self, state):
        return False
