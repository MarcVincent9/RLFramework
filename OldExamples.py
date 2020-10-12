# -*- coding: utf-8 -*-

from MarkovTypes import *
import numpy as np
import collections


class OldTicTacToe(SequentialDeterministicGame):

    EMPTY = '_'
    PLAYERS = 'XO'
    State = collections.namedtuple("State", ["board", "player"])

    def __init__(self, n=3, d=3):
        self.n = n
        self.d = d
        self.initial = self.State(self.EMPTY * self.n ** 2, 0)
        self.win, self.draw, self.state_set = set(), set(), {self.initial}
        self.subsequent_states(self.initial)

    def states(self):
        return list(self.state_set)

    def sample_initial_state(self):
        return self.initial

    def is_state_terminal(self, state):
        return state in self.win or state in self.draw

    def agents(self):
        return [0, 1]

    def actions(self, state):
        return [i for i, c in enumerate(state.board) if c == self.EMPTY]

    def certain_transition(self, state, action):
        board, player = state
        other = (player + 1) % 2
        next_board = list(board)
        next_board[action] = self.PLAYERS[player]
        next_state = self.State(''.join(next_board), other)
        rewards = [0, 0]
        if next_state in self.win:
            rewards[player] = 1
            rewards[other] = -1
        elif next_state in self.draw:
            rewards = [.5, .5]
        return (next_state, *rewards)

    def turn(self, state):
        return state.player

    def gamma(self, state):
        return .999

    def to_string(self, state):
        board, player = state
        res = ""
        for i in range(self.n):
            res += ' '.join(board[i * self.n: (i+1) * self.n]) + '\n'
        return res + '(%s)'%self.PLAYERS[player]

    def subsequent_states(self, state):
        board, player = state
        target = self.PLAYERS[player] * self.d
        actions = self.multi_actions(state, player)
        if not actions:
            self.draw.add(state)
        else:
            for i in actions:
                next_board = list(board)
                next_board[i] = self.PLAYERS[player]
                next_state = self.State(''.join(next_board), (player + 1) % 2)
                if next_state not in self.state_set:
                    self.state_set.add(next_state)
                    npboard = np.resize(next_board, (self.n, self.n))
                    action = (i // self.n, i % self.n)
                    for line in [npboard[action[0], max(0, action[1] - self.d + 1): action[1] + self.d],
                                 npboard[max(0, action[0] - self.d + 1): action[0] + self.d, action[1]],
                                 np.diag(npboard, k=action[1] - action[0])[max(0, min(action) - self.d + 1): min(action) + self.d],
                                 np.diag(np.fliplr(npboard), k=self.n - 1 - action[1] - action[0])[max(0, min(action[0], self.n - 1 - action[1]) - self.d + 1): min(action) + self.d]]:
                        if target in ''.join(line):
                            self.win.add(next_state)
                            break
                    else:
                        self.subsequent_states(next_state)


class OldConnectN(SequentialDeterministicGame):

    EMPTY = '_'
    PLAYERS = 'RW'
    State = collections.namedtuple("State", ["board", "player"])

    def __init__(self, width=7, height=6, n=4):
        self.n = n
        self.width = width
        self.height = height
        self.initial = self.State((self.EMPTY * self.height,) * self.width, 0)
        self.win, self.draw, self.state_set = set(), set(), {self.initial}

    def states(self):
        if len(self.state_set) == 1:
            self.subsequent_states(self.initial)
        return list(self.state_set)

    def sample_initial_state(self):
        return self.initial

    def is_state_terminal(self, state):
        return self.is_state_win(state) or self.is_state_draw(state)

    def agents(self):
        return [0, 1]

    def actions(self, state):
        return [i for i, column in enumerate(reversed(state.board)) if column.find(self.EMPTY) >= 0]

    def certain_transition(self, state, action):
        next_state, i_column, top = self.next_state(state, action)
        rewards = [0, 0]
        if self.is_move_win(next_state, i_column, top):
            rewards[state.player] = 1
            rewards[next_state.player] = -1
        elif self.is_state_draw(next_state):
            rewards = [.5, .5]
        return (next_state, *rewards)

    def turn(self, state):
        return state.player

    def gamma(self, state):
        return .999

    def to_string(self, state):
        res = ""
        for h in reversed(range(self.height)):
            for column in reversed(state.board):
                res += column[h] + ' '
            res += '\n'
        return res + "(%s)" % state.player

    def next_state(self, state, action):
        i_column = self.width - 1 - action
        next_board = list(state.board)
        top = next_board[i_column].find(self.EMPTY)
        next_board[i_column] = list(next_board[i_column])
        next_board[i_column][top] = self.PLAYERS[state.player]
        next_board[i_column] = ''.join(next_board[i_column])
        next_state = self.State(tuple(next_board), (state.player + 1) % 2)
        return next_state, i_column, top

    def is_move_win(self, next_state, i_column, top):
        if next_state in self.win:
            return True
        target = self.PLAYERS[(next_state.player + 1) % 2] * self.n
        npboard = np.resize(list(''.join(next_state.board)), (self.width, self.height)).T
        action = (top, i_column)
        for line in [npboard[action[0], max(0, action[1] - self.n + 1): action[1] + self.n],
                     npboard[max(0, action[0] - self.n + 1): action[0] + 1, action[1]],
                     np.diag(npboard, k=action[1] - action[0])[
                     max(0, min(action) - self.n + 1): min(action) + self.n],
                     np.diag(np.fliplr(npboard), k=self.width - 1 - action[1] - action[0])[
                     max(0, min(action[0], self.width - 1 - action[1]) - self.n + 1): min(action) + self.n]]:
            if target in ''.join(line):
                self.win.add(next_state)
                return True
        return False

    def is_state_win(self, state):
        if state in self.win:
            return True
        for action in range(self.width):
            i_column = self.width - 1 - action
            top = state.board[i_column].find(self.EMPTY)
            top = top - 1 if top >= 0 else top
            if self.is_move_win(state, i_column, top):
                return True
        return False

    def is_state_draw(self, state):
        """assumes state is not a win..."""
        if state in self.draw:
            return True
        if ''.join(state.board).find(self.EMPTY) == -1:
            self.draw.add(state)
            return True
        return False

    def subsequent_states(self, state):
        if not self.is_state_draw(state):
            for action in self.actions(state):
                next_state, i_column, top = self.next_state(state, action)
                if next_state not in self.state_set:
                    self.state_set.add(next_state)
                    if not self.is_move_win(next_state, i_column, top):
                        self.subsequent_states(next_state)
