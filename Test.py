# -*- coding: utf-8 -*-

from MonoAgentLearning import *
from SequentialDualAgentLearning import *
from MultiAgentLearning import *
from Examples import *
from util import roulette

bootstrap = 4
nb_episodes = 10000

"""
# Random walk

rw = RandomWalk(51, leftmost_reward=-1)
random_policy = {state: {action: 1 / len(rw.actions(state)) for action in rw.actions(state)} for state in rw.states()}

learningV0 = MDPLearning(rw)
learningV0.policy_evaluation(random_policy, theta=.000000001)

learningV1 = MDPLearning(rw)
learningV1.one_step_td(random_policy, time_limit=3)

learningV2 = MDPLearning(rw)
learningV2.n_step_td(random_policy, bootstrapping_depth=bootstrap, time_limit=3)


# Gridworld

size = 10
win = (size // 2, 3 * size // 4)
ini = (size // 2, size // 4)
gw = GridWorld(nb_lines=size, nb_columns=size, initial=[ini], win=[win])


def correctness(policy):
    res = 0
    n = 0
    for l in range(gw.nb_lines):
        for c in range(gw.nb_columns):
            try:
                direction = np.array(win) - np.array([l, c])
                if policy[l, c][0] * direction[0] > 0 or policy[l, c][1] * direction[1] > 0:
                    res+=1
                n += 1
            except KeyError:
                pass
    return res/n


learning = MDPLearning(gw)
#learning.target_policy_expectation = learning.greedy_sarsa_expectation
learning.policy_iteration()
print("Policy iteration")
gw.print_policy(learning.optimal_policy)

learning.value_iteration()
learning.policy_improvement()
print("Value iteration")
gw.print_policy(learning.optimal_policy)

learning.one_step_action_learning(episode_limit=nb_episodes, behavior_epsilon=.1, target_epsilon=.05)
learning.get_policy()
print("One-step Sarsa")
gw.print_policy(learning.optimal_policy)
print(correctness(learning.optimal_policy))

learning.n_step_action_learning(bootstrapping_depth=bootstrap, episode_limit=nb_episodes,
                                behavior_epsilon=.1, target_epsilon=.05)
learning.get_policy()
print("%d-step Sarsa"%bootstrap)
gw.print_policy(learning.optimal_policy)
print(correctness(learning.optimal_policy))


# RL on Tictactoe + human assessment

ttt = OldTicTacToe()
sdglearning = SDGLearning(ttt)
sdglearning.one_step_afterstate_learning(time_limit=60)
sdglearning.players[0].get_policy()
p = sdglearning.players[0].policy

eg = ExperienceGenerator(ttt)
class FixedPolicyPlayer(ArtificialPlayer):
    def target_policy(self, state):
        return roulette(p[state])
eg.add_player(0, FixedPolicyPlayer, learning=False)
eg.add_player(1, HumanPlayer)
eg.generate_experience(verbose=True)


# same with more generic code

ttt = TicTacToe()
eg = ExperienceGenerator(ttt)
eg.add_player(0, SDGTabularArtificialPlayer)
eg.add_player(1, SDGTabularArtificialPlayer)
eg.generate_experience(time_limit=30)
eg.players[0].learning = False
eg.add_player(1, HumanPlayer)
eg.generate_experience(verbose=True)
"""


class A:
    
    def __init__(self, deterministic=False):
        if deterministic:
            def deter(self):
                return 2
            setattr(A, "deter", deter)
    
    def expect(self):
        return 1
    