from typing import Tuple
import numpy as np
from numpy.linalg import inv
import numpy.random as rn
import matplotlib.pyplot as plt
import pickle
import cvxpy as cp
from tqdm import tqdm


## Define the gridworld MDP class
class Gridworld(object):
    """
    Gridworld MDP.
    """
    f1 = np.array([
            [0,0,0,0,0,0,0,0,0,0], 
            [0,0,0,0,0,0,0,0,0,0], 
            [0,0,0,0,0,-10,-10,0,0,0], 
            [0,0,0,0,0,-10,-10,0,0,0],
            [0,-10,-10,0,0,0,0,0,0,0],
            [0,-10,-10,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,-10,-10,0,0,0,0,0,0],
            [0,0,-10,-10,0,0,0,0,0,1]
        ])

    f2 = np.array([
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,-100,-100,-100,0,0,0],
        [0,0,0,0,-100,0,-100,0,0,0],
        [0,0,0,0,-100,0,-100,-100,-100,0],
        [0,0,0,0,-100,0,0,0,-100,0],
        [0,0,0,0,-100,0,0,0,-100,0],
        [0,0,0,0,-100,0,0,0,-100,0],
        [0,0,0,0,0,0,-100,-100,-100,0],
        [0,0,0,0,0,0,-100,0,0,0],
        [0,0,0,0,0,0,0,0,0,10]
    ])
    
    def __init__(self, grid_size, wind, discount, reward_function):
        """
        grid_size: Grid size. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        -> Gridworld
        """

        self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        self.n_actions = len(self.actions)
        self.n_states = grid_size**2
        self.grid_size = grid_size
        self.wind = wind
        self.discount = discount
        self.reward_function = reward_function

        # Preconstruct the transition probability array.
        self.transition_probability = np.array(
            [[[self._transition_probability(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])

    def __str__(self):
        return "Gridworld({}, {}, {})".format(self.grid_size, self.wind,
                                              self.discount)
    
    def int_to_point(self, i):
        """
        Convert a state int into the corresponding coordinate.

        i: State int.
        -> (x, y) int tuple.
        """

        return (i % self.grid_size, i // self.grid_size)

    def point_to_int(self, p):
        """
        Convert a coordinate into the corresponding state int.

        p: (x, y) tuple.
        -> State int.
        """

        return int(p[0] + p[1]*self.grid_size)

    def neighbouring(self, i, k):
        """
        Get whether two points neighbour each other. Also returns true if they
        are the same point.

        i: (x, y) int tuple.
        k: (x, y) int tuple.
        -> bool.
        """

        return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1

    def moving_off_grid(self,xi,xj,yi,yj,k):
        return k < 0 or k > 99 or \
            (xi + xj < 0 or xi + xj > 9) or \
            (yi + yj < 0 or yi + yj > 9)

    def _transition_probability(self, i, j, k):
        """
        Get the probability of transitioning from state i to state k given
        action j.

        i: State int.
        j: Action int.
        k: State int.
        -> p(s_k | s_i, a_j)
        """
        corner_points = {0, 9, 90, 99}
        edge_points = {1,2,3,4,5,6,7,8,
                        10,19,20,29,30,39,40,49,50,59,60,69,70,79,80,89,
                        91,92,93,94,95,96,97,98
                    }

        xi, yi = self.int_to_point(i)
        xj, yj = self.actions[j]
        xk, yk = self.int_to_point(k)
        
        if not self.neighbouring((xi, yi), (xk, yk)):
            return 0

        # Is k the intended state to move to?
        if (xi + xj, yi + yj) == (xk, yk):
            return 1 - self.wind*(3/4)

        # If these are not the same point, then we can move there by wind.
        if (xi, yi) != (xk, yk):
            return self.wind / 4
        
        # Same point, corner?
        if i in corner_points:
            if self.moving_off_grid(xi,xj,yi,yj,k):
                return 1 - self.wind * 1/2
            else: # blow off the grid by wind.
                return self.wind * 1/2
        elif i in edge_points:
            if self.moving_off_grid(xi,xj,yi,yj,k):
                return 1 - self.wind * 3/4
            else: # blow off the grid by wind.
                return self.wind * 1/4
        else:
            return 0

    def reward(self, state_int):
        """
        Reward for being in state state_int.

        state_int: State integer. int.
        -> Reward.
        """
        #look at figure 6,7 to return a reward at the given state. 
        xi, yi = self.int_to_point(state_int)
        if self.reward_function == 1:
            reward_val = self.f1[xi,yi]
        else:
            reward_val = self.f2[xi,yi]
        
        return reward_val

    @property
    def reward_array(self):
        if self.reward_function == 1:
            return self.f1
        else:
            return self.f2