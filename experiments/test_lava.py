from __future__ import division
from __future__ import print_function
from operator import ne

import random
import argparse

import numpy as np

from mcts.mcts import MCTS
from mcts.states import LavaCrossingWorld as World
from mcts.states import LavaCrossingState as State
from mcts.states import LavaCrossingAction as Action
from mcts.graph import ActionNode, StateNode
from mcts.tree_policies import UCB1
from mcts.default_policies import immediate_reward, random_terminal_roll_out
from mcts.backups import monte_carlo
import numpy as np


__author__ = 'lemon'

from gym_minigrid.envs import LavaCrossingS9N2Env, MyLavaCrossingS9N2Env
from gym_minigrid.wrappers import FullyObsWrapper


def run_experiment(c, mc_n, runs, steps):    
    env = MyLavaCrossingS9N2Env()
    all_rewards = []
    for iter in range(runs):
        print("{} out of {} runs.".format(iter, runs))

        ep_reward = []
        grid = env.reset()
        grid = np.array(grid).reshape(9,9)
        # grid = np.array([[ 2,  2,  2,  2,  2,  2,  2,  2,  2],
        #         [ 2, 10,  9,  1,  1,  1,  1,  1,  2],
        #         [ 2,  1,  1,  1,  1,  1,  1,  1,  2],
        #         [ 2,  1,  9,  1,  1,  1,  1,  1,  2],
        #         [ 2,  1,  9,  1,  1,  1,  1,  1,  2],
        #         [ 2,  1,  9,  1,  1,  1,  1,  1,  2],
        #         [ 2,  9,  9,  9,  9,  9,  1,  9,  2],
        #         [ 2,  1,  9,  1,  1,  1,  1,  8,  2],
        #         [ 2,  2,  2,  2,  2,  2,  2,  2,  2]])
        done = False
        while not done:
            y,x = np.where(grid==10)
            pos = [x[0],y[0]] 
            world = World(grid)

            root_state = State(pos, world, max_steps=steps)
            next_state = StateNode(None, root_state)

            mcts = MCTS(tree_policy=UCB1(c=c), 
            default_policy=immediate_reward,
            backup=monte_carlo)
            best_action = mcts(next_state, n=mc_n)

            # next_s = next_state.children[best_action].sample_state()
            # print(next_s.state.pos)

            # tmpt_grid = next_s.state.world.grid.copy()
            # _x,_y = next_s.state.pos
            # tmpt_grid[_y, _x] = 10
            # print(tmpt_grid)
            # input()

            # if (next_s.state.is_terminal()):
            #     break

            # next_state = next_s
            # next_state.parent = None
            grid, reward, done, _ = env.step(best_action.action)
            grid = np.array(grid).reshape(9,9)
            # print(best_action.action)
            print("\rSteps: {}".format(env.step_count), end="", flush=True)
            # print(grid)
            ep_reward.append(reward)
        print("")
        all_rewards.append(np.mean(ep_reward))
    # print("All rewards stat:\n mean:{}, std:{}".format(np.mean(all_rewards), np.std(all_rewards)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiment for UCT with '
                                                 'intrinsic motivation.')
    parser.add_argument('--mcsamples', '-m', type=int, default=500,
                        help='How many monte carlo runs should be made.')
    parser.add_argument('--runs', '-r', type=int, default=10,
                        help='How many runs should be made.')
    parser.add_argument('--steps', '-s', type=int, default=100,
                        help="Maximum number of steps performed.")
    parser.add_argument('--uct_c', '-c', type=float, default=5,
                        help='The UCT parameter Cp.')

    args = parser.parse_args()
    run_experiment(mc_n=args.mcsamples, runs=args.runs, steps=args.steps,
                   c=args.uct_c)


