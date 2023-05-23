import sys
from typing import Tuple
from statistics import mode
import numpy as np
from numpy.linalg import inv
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import cvxpy as cp
from cvxopt import matrix, solvers
from proj3_gridworld import Gridworld


def accuracy(expected_policy, test_policy):
    return np.sum(expected_policy == test_policy) / len(expected_policy)


class Agent:
    """
    Agent class for the gridworld MDP.

    The agent is initialized with a gridworld MDP and an initial policy.
    The agent can then update its policy using value iteration.

    The agent can:
    - generate trajectories from its policy
    - evaluate the performance of its policy
    - generate a heat map of the value function
    - generate a heat map of the policy
    - generate a heat map of the ground truth reward
    - generate a heat map of the reward function

    Variables
    ---------
    gw: Gridworld
        The gridworld MDP.
    value_matrix: np.ndarray
        The value function for the agent's policy.
    optimal_policy: np.ndarray
        The optimal policy for the agent's MDP.
    action_matrix: np.ndarray
        The optimal policy for the agent's MDP in matrix form.

    Methods
    -------
    optimal_value
        Compute the optimal value function for the agent's MDP.
    find_policy
        Compute the optimal policy for the agent's MDP.
    """

    def __init__(self, gw: Gridworld):
        gw.transition_probability = np.transpose(gw.transition_probability, (1, 0, 2))
        self.gw = gw
        self.optimal_policy, v = Agent.find_policy(
            gw.n_states,
            gw.n_actions,
            gw.transition_probability,
            gw.reward_array.T.flatten(),
            gw.discount,
        )
        self.values = np.array(v)
        self.reward_matrix = gw.reward_array
        self.rewards = gw.reward_array.T.flatten()
        self.value_matrix = v.reshape((gw.grid_size, gw.grid_size)).T
        self.action_matrix = self.optimal_policy.reshape(
            (gw.grid_size, gw.grid_size)
        ).T.astype(int)
        
        # Setup IRL problem, the problem is run in run_irl
        def action_value(a, s):
            a1 = self.optimal_policy[s]
            a = gw.transition_probability[a1, s] - gw.transition_probability[a, s]
            b = inv(np.eye(gw.n_states) - gw.discount * gw.transition_probability[a1])
            return a @ b

        R_max = np.abs(gw.reward_array).max()
        self.x = cp.Variable((3 * gw.n_states, 1))  # R, t, u
        n_s = gw.n_states
        n_a = gw.n_actions
        I = np.eye(n_s)
        Z = np.zeros((n_s, n_s))

        D1 = np.block([
            [ I, Z, -I],
            [-I, Z, -I],
            [ I, Z,  Z],
            [-I, Z,  Z],
        ])
        b1 = np.zeros((4*n_s, 1))
        b1[2*n_s:] = R_max

        reward_factor = np.empty(((n_a-1)*n_s, n_s))
        i = 0
        for a in range(n_a):
            for s in range(n_s):
                if a != self.optimal_policy[s]:
                    reward_factor[i] = action_value(a, s)
                    i += 1
        D2 = np.block([
            [-reward_factor, np.zeros(((n_a-1)*n_s, 2*n_s))],
            [-reward_factor[:n_s], I, Z],
            [-reward_factor[n_s:2*n_s], I, Z],
            [-reward_factor[2*n_s:], I, Z],
        ])
        b2 = np.zeros((((n_a-1) + 3)*n_s, 1))

        self.D = np.vstack((D1, D2))
        self.b = np.vstack((b1, b2))

        self.constraints = [self.D @ self.x <= self.b]

    ## Implementing the function for computing the optimal policy.
    ## The function takes as input the MDP and outputs a
    ## deterministic policy, which is an array of actions.
    ## The i^th entry in the array corresponds to the
    ## optimal action to take at the i^th state.
    @staticmethod
    def find_policy(
        n_states,
        n_actions,
        transition_probabilities,
        reward,
        discount,
        threshold=1e-2,
        values=None,
        stochastic=False,
    ):
        """
        Find the optimal value function.

        n_states: Number of states. int.
        n_actions: Number of actions. int.
        transition_probabilities: Function taking (state, action, state) to
            transition probabilities.
        reward: Vector of rewards for each state.
        discount: MDP discount factor. float.
        threshold: Convergence threshold, default 1e-2. float.

        Returns
        -------
        policy: Array of actions for each state
        values: Array of values for each state
        """
        # Estimation
        if values is None:
            values = np.zeros(n_states)
            delta = 1 + threshold
            while delta > threshold:
                delta = 0
                for state in range(n_states):
                    v_old = values[state]
                    values[state] = np.max(
                        transition_probabilities[:, state, :]
                        @ (reward + discount * values)
                    )
                    delta = max(delta, abs(v_old - values[state]))

        # Computation:
        policy = np.argmax(
            transition_probabilities @ (reward + discount * values), axis=0
        )
        return policy, values

    def run_irl(self, l1: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Returns policy found by inverse reinforcement learning and its accuracy when compared to the optimal policy

        Returns
        -------
        policy: np.ndarray
            Array of actions for each state
        values: np.ndarray
            Array of values for each state
        acc: float
            Accuracy of the policy found by IRL
        """
        n_s = self.gw.n_states
        c = np.zeros((3*n_s))
        c[n_s:2*n_s] = 1
        c[2*n_s:] = -l1
        cp.Problem(
            cp.Maximize(c @ self.x), self.constraints
        ).solve()
        if self.x.value is None:
            raise ValueError("The problem is infeasible")
        R = self.x.value[:n_s, 0]

        # solve with cvxopt, doesn't do any better than cvxpy
        # sol = solvers.lp(matrix(-c) ,matrix(self.D), matrix(self.b))
        # R = np.array(sol['x'][:n_s]).flatten()

        irl_policy, values = Agent.find_policy(
            self.gw.n_states,
            self.gw.n_actions,
            self.gw.transition_probability,
            R,
            self.gw.discount,
        )
        acc = accuracy(self.optimal_policy, irl_policy)

        self.irl_reward_function = R
        self.irl_values = values
        self.irl_policy = irl_policy
        self.irl_reward_matrix = self.irl_reward_function.reshape(
            (self.gw.grid_size, self.gw.grid_size)
        ).T
        self.irl_value_matrix = values.reshape((self.gw.grid_size, self.gw.grid_size)).T
        self.irl_action_matrix = (
            irl_policy.reshape((self.gw.grid_size, self.gw.grid_size)).astype(int).T
        )

        return irl_policy, values, acc

    ## Function for plotting the matrix values
    def plot_value_matrix(self, policy_type="original"):
        """
        policy_type: {'original', 'irl'}
        """
        if policy_type == "original":
            matrix = self.value_matrix
        elif policy_type == "irl":
            matrix = self.irl_value_matrix
        fig, ax = plt.subplots()
        num_rows = len(matrix)
        min_val, max_val = 0, num_rows

        for i in range(num_rows):
            for j in range(num_rows):
                c = matrix[i][j]
                ax.text(j + 0.5, i + 0.5, "{:.1f}".format(c), va="center", ha="center")

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(max_val, min_val)
        ax.set_xticks(np.arange(max_val))
        ax.set_yticks(np.arange(max_val))
        ax.xaxis.tick_top()
        ax.grid()
        plt.show()
        plt.close()

    def plot_reward_heatmap(self, policy_type="original"):
        """
        policy_type: {'original', 'irl'}
        """
        if policy_type == "original":
            matrix = self.reward_matrix
        elif policy_type == "irl":
            matrix = self.irl_reward_matrix
        ## For visualization generating the heat map of the optimal state values
        plt.pcolor(np.flipud(matrix))
        # plt.pcolor(matrix)
        plt.colorbar()
        plt.axis("off")
        plt.title(
            f"Optimal reward values for Reward function {self.gw.reward_function}"
        )
        plt.show()

    def plot_value_heatmap(self, policy_type="original"):
        """
        policy_type: {'original', 'irl'}
        """
        if policy_type == "original":
            matrix = self.value_matrix
        elif policy_type == "irl":
            matrix = self.irl_value_matrix
        ## For visualization generating the heat map of the optimal state values
        plt.pcolor(np.flipud(matrix))
        plt.colorbar()
        plt.axis("off")
        plt.title(f"Optimal state values for Reward function {self.gw.reward_function}")
        plt.show()

    def plot_arrow(self, policy_type="original"):
        """Function for plotting the optimal actions at each state in the grid
        The function takes as input the matrix containing optimal actions
        and plots the actions for each state on the grid
        """
        if policy_type == "original":
            matrix = self.action_matrix
        elif policy_type == "irl":
            matrix = self.irl_action_matrix
        _, ax = plt.subplots()
        num_rows = len(matrix)
        min_val, max_val = 0, num_rows
        arrow_dict = {0: "↓", 1: "→", 2: "↑", 3: "←"}

        for i in range(num_rows):
            for j in range(num_rows):
                ax.text(
                    j + 0.5, i + 0.5, arrow_dict[matrix[i][j]], va="center", ha="center"
                )

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(max_val, min_val)
        ax.set_xticks(np.arange(max_val))
        ax.set_yticks(np.arange(max_val))
        ax.xaxis.tick_top()
        ax.grid()
