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

        '''
        # setup Inverse reinforcement learning cvxpy code
        # a1 = mode(self.optimal_policy)
        # R_max = np.abs(gw.reward_array).max()
        # P_a1 = gw.transition_probability[a1]
        # self.R = cp.Variable((gw.n_states, 1))
        # self.t = cp.Variable((gw.n_states, 1))
        # self.u = cp.Variable((gw.n_states, 1), nonneg=True)
        # self.constraints = [
        #     self.R <= self.u,
        #     self.R >= -self.u,
        #     self.R <= R_max,
        #     self.R >= -R_max,
        # ]
        # for a in range(gw.n_actions):
        #     if a != a1:
        #         P_a = gw.transition_probability[a]
        #         self.constraints.append((P_a1 - P_a) @ inv(np.eye(gw.n_states) - gw.discount * P_a1) @ self.R >= 0)
        #         self.constraints.append((P_a1 - P_a) @ inv(np.eye(gw.n_states) - gw.discount * P_a1) @ self.R >= self.t)
        '''
        # setup Inverse reinforcement learning cvxpy code
        def action_value(a, s):
            a1 = self.optimal_policy[s]
            a = gw.transition_probability[a1, s] - gw.transition_probability[a, s]
            b = inv(np.eye(gw.n_states) - gw.discount * gw.transition_probability[a1])
            return a @ b

        R_max = np.abs(gw.reward_array).max()
        self.R = cp.Variable((gw.n_states, 1))
        self.t = cp.Variable((gw.n_states, 1))
        self.u = cp.Variable((gw.n_states, 1), nonneg=True)
        self.constraints = [
            self.R <= self.u,
            self.R >= -self.u,
            self.R <= R_max,
            self.R >= -R_max,
        ]
        i = 0
        tmp = np.empty(((gw.n_actions-1)*gw.n_states, gw.n_states))
        for a in range(gw.n_actions):
            tmp2 = np.zeros((gw.n_states, gw.n_states))
            for s in range(gw.n_states):
                if a != self.optimal_policy[s]:
                    tmp[i] = action_value(a, s)
                    tmp2[s] = tmp[i]
                    i += 1
        self.constraints.append(tmp @ self.R >= 0)
        self.constraints.append(tmp[:gw.n_states] @ self.R >= self.t)
        self.constraints.append(tmp[gw.n_states:2*gw.n_states] @ self.R >= self.t)
        self.constraints.append(tmp[2*gw.n_states:] @ self.R >= self.t)

        # n = gw.n_states
        # self.x = cp.Variable((3*n, 1)) # R, t, u
        # P = gw.transition_probability
        # D = np.zeros((10*n, 3*n))
        # D[:n, :n]       =  np.eye(n)
        # D[:n, 2*n:]     = -np.eye(n)
        # D[n:2*n, :n]    = -np.eye(n)
        # D[n:2*n, 2*n:]  =  np.eye(n)
        # D[2*n:3*n, :n]  =  np.eye(n)
        # D[3*n:4*n, :n]  = -np.eye(n)
        # i = 0
        # for a in range(gw.n_actions):
        #     if a != a1:
        #         D[4*n+i*n:4*n+(i+1)*n, :n] = -(P[a1] - P[a]) @ inv(np.eye(n) - gw.discount * P[a1])
        #         i += 1
        # i = 0
        # for a in range(gw.n_actions):
        #     if a != a1:
        #         D[7*n+i*n:7*n+(i+1)*n, :n] = -(P[a1] - P[a]) @ inv(np.eye(n) - gw.discount * P[a1])
        #         D[7*n+i*n:7*n+(i+1)*n, n:2*n] = np.eye(n)
        #         i += 1
        # b = np.zeros((10*n, 1))
        # b[2*n:4*n] = R_max
        # self.constraints = [D @ self.x <= b]
        # self.D = D
        # self.b = b

    def new_irl(self, lambda_val: float = 0):
        # x is of dims 3*n, 1, holds R, t, u
        R_max = np.abs(self.gw.reward_array).max()
        n = self.gw.n_states
        P = self.gw.transition_probability

        # Calculate D
        a1 = 0
        D = np.zeros((10 * n, 3 * n))
        tmp = inv(np.eye(n) - self.gw.discount * P[a1])
        # Constraint 1
        D[:n, :n] = -(P[a1] - P[1]) @ tmp
        D[:n, n : 2 * n] = np.eye(n)
        D[n : 2 * n, :n] = -(P[a1] - P[2]) @ tmp
        D[n : 2 * n, n : 2 * n] = np.eye(n)
        D[2 * n : 3 * n, :n] = -(P[a1] - P[3]) @ tmp
        D[2 * n : 3 * n, n : 2 * n] = np.eye(n)

        # Constraint 2
        D[3 * n : 4 * n, :n] = -(P[a1] - P[1]) @ tmp
        D[4 * n : 5 * n, :n] = -(P[a1] - P[2]) @ tmp
        D[5 * n : 6 * n, :n] = -(P[a1] - P[3]) @ tmp

        # Constraint 3
        D[6 * n : 7 * n, :n] = np.eye(n)
        D[6 * n : 7 * n, 2 * n :] = -np.eye(n)
        D[7 * n : 8 * n, :n] = -np.eye(n)
        D[7 * n : 8 * n, 2 * n :] = -np.eye(n)

        # Constraint 4
        D[8 * n : 9 * n, :n] = np.eye(n)
        D[9 * n :, :n] = -np.eye(n)

        # Calculate b
        b = np.zeros((10 * n))
        b[8 * n :] = R_max

        # Calculate c
        c = np.zeros((3 * n))
        c[n:] = -1
        c[2 * n :] = lambda_val

        # solve linear program
        sol = solvers.lp(matrix(c), matrix(A), matrix(b))
        R = np.array(sol["x"][:n])

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

    def run_irl(self, lambda_val: float) -> Tuple[np.ndarray, np.ndarray, float]:
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
        cp.Problem(
            cp.Maximize(cp.sum(self.t) - lambda_val * cp.sum(self.u)), self.constraints
        ).solve()
        if self.R.value is None:
            raise ValueError("The problem is infeasible")
        R = self.R.value[:, 0]

        # Trying a new cvxpy method
        # c = np.zeros((3*self.gw.n_states, 1))
        # c[self.gw.n_states:2*self.gw.n_states] = 1
        # c[2*self.gw.n_states:] = -lambda_val
        # cp.Problem(cp.Maximize(c.T @ self.x), self.constraints).solve()
        # if self.x.value is None:
        #     raise ValueError('The problem is infeasible')
        # R = self.x.value[:self.gw.n_states, 0]

        # # Trying a new scipy method
        # res = linprog(-c[:,0], A_ub=self.D, b_ub=self.b)
        # R = res.x[:self.gw.n_states]

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
