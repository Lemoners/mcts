from __future__ import division
from __future__ import print_function

import numpy as np

""" 
---width----> X
|
|
height
|
|
V
Y
"""

ACTION_TO_DIR = {
        0: [0, -1],
        1: [0, 1],
        2: [-1, 0],
        3: [1, 0] 
    }
OBJECT_TO_IDX = {
    'unseen'        : 0,
    'empty'         : 1,
    'wall'          : 2,
    'floor'         : 3,
    'door'          : 4,
    'key'           : 5,
    'ball'          : 6,
    'box'           : 7,
    'goal'          : 8,
    'lava'          : 9,
    'agent'         : 10,
}
IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))


class LavaCrossingAction(object):

    def __init__(self, action):
        """
        Args:
            action (int): integer from 0-3
        """
        self.action = action
        self._hash = int(action)

    def __hash__(self):
        return int(self._hash)

    def __eq__(self, other):
        if other == None:
            return False
        else:
            return (self.action == other.action)

    def __str__(self):
        return str(self.action)

    def __repr__(self):
        return str(self.action)


class LavaCrossingWorld(object):
    def __init__(self, grid):
        self.grid = grid.copy()
        self.height, self.width = grid.shape

        g_y, g_x = np.where(grid==OBJECT_TO_IDX['goal'])
        self.goal_pos = [g_x[0], g_y[0]]

        y, x = np.where(grid==OBJECT_TO_IDX['agent'])
        self.grid[y,x] = OBJECT_TO_IDX['empty']


class LavaCrossingState(object):
    def __init__(self, pos, world, steps=0, done=False, max_steps=500):
        """
        Args:
            pos ([x, y])
            world (np.array): World, without agent.
            done (bool, optional): done. Defaults to False.
        """
        self.world = world
        self.height, self.width = world.height, world.width
        assert OBJECT_TO_IDX['agent'] not in self.world.grid
        self.pos = pos
        self.goal_pos = self.world.goal_pos
        self.done = done
        self.steps = steps
        self.max_steps = max_steps


        if self.steps >= self.max_steps:
            self.done = True

        self.actions = [LavaCrossingAction(i) for i in range(4)]

    def perform(self, action):
        delta = ACTION_TO_DIR[action.action]
        new_x = np.clip(self.pos[0]+delta[0], 0, self.width-1)
        new_y = np.clip(self.pos[1]+delta[1], 0, self.height-1)
        done = False
        # Do I hit the wall or lava
        if self.world.grid[new_y, new_x] == OBJECT_TO_IDX['wall']:
            pos = self.pos
        elif self.world.grid[new_y, new_x] == OBJECT_TO_IDX['lava']:
            pos = [new_x, new_y]
            done = True
        elif self.world.grid[new_y, new_x] == OBJECT_TO_IDX['empty']:
            pos = [new_x, new_y]
        elif self.world.grid[new_y, new_x] == OBJECT_TO_IDX['goal']:
            pos = [new_x, new_y]
            done = True
        else:
            raise Exception("Not implement item with idx {}.".format(self.world.grid[new_y, new_x]))

        return LavaCrossingState(pos, self.world, steps=self.steps+1, done=done, max_steps=self.max_steps)

    def is_terminal(self):
        return self.done

    def __eq__(self, other):
        if other == None:
            return False
        else:
            return (self.pos == other.pos)

    def __hash__(self):
        return int(str(self.pos[0])+str(self.pos[1]))

    def __str__(self):
        return str(self.pos)

    def __repr__(self):
        return str(self.pos)

    def reward(self, parent, action):
        reward = 0
        x,y = self.pos
        if (self.pos == self.goal_pos):
            reward = 1000
        elif (self.world.grid[y,x] == OBJECT_TO_IDX['lava']):
            reward = -1000
        elif (self.pos == parent.pos):
            reward = -0.1
        else:
            reward = -0.05*np.linalg.norm([self.pos[0]-self.goal_pos[0], self.pos[1]-self.goal_pos[1]], ord=2)
            # reward = 0
        return reward
