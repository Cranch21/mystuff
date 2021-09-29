import time
import numpy as np
import itertools
import game_env

from game_state import GameState

"""
solution.py

Template file for you to implement your solution to Assignment 2.

You must implement the following method stubs, which will be invoked by the simulator during testing:
    __init__(game_env)
    plan_offline()
    select_action()
    
To ensure compatibility with the autograder, please avoid using try-except blocks for Exception or OSError exception
types. Try-except blocks with concrete exception types other than OSError (e.g. try: ... except ValueError) are allowed.

COMP3702 2021 Assignment 2 Support Code

Last updated by njc 02/09/21
"""


# https://ai-boson.github.io/mcts/
class MonteCarloTreeSearchNode():
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        # self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        return


# TODO need to adjust this from tutorial to this assignment
class ValueIteration:

    def __init__(self, grid, game_env):

        self.grid = grid
        self.environment = game_env
        self.values = {}
        self.policy = {}
        self.exit_row = self.environment.exit_row
        self.exit_col = self.environment.exit_col
        self.exit_gems = tuple([1 for _ in self.environment.get_init_state().gem_status])
        self.exit_state = GameState(self.exit_row, self.exit_col, self.exit_gems)
        self.goal = 0
        self.lava = -100
        # ToDo check if this needs to be in value implementation
        '''
        if s.row == self.environment.GameEnv.exit_row and s.column == self.environment.GameEnv.exit_col:
            reward = 100
        '''
        # ToDo check if this needs to be in policy implementation
        for r, row in enumerate(game_env.grid_data):
            for c, column in enumerate(row):

                if column != 'X' and column != '*':
                    for gem in self.find_gem_possibility():
                        state = GameState(r, c, gem)

                        self.values[state] = 0
                        self.policy[state] = self.environment.WALK_RIGHT
                elif column == '*':
                    self.values[state] = self.grid.game_over_penalty
        # print('sup', self.values)
        # print(self.policy)
        self.epsilon = 0.001
        self.converged = False
        self.actions = game_env.ACTIONS

    # adapted from
    # https://www.techiedelight.com/generate-permutations-list-python/#:~:text=In%20Python%2C%20you%20can%20use%20the%20in
    # -built%20module,and%20hence%20generates%20all%20possible%20permutations.%201%202
    def find_gem_possibility(self):

        permutations = [tuple(x) for x in
                        itertools.product([1, 0], repeat=len(self.environment.get_init_state().gem_status))]
        return permutations

    def get_action_name(self, action):
        return action

    def next_iteration(self):
        """
        Write code here to imlpement the VI value update
        Iterate over self.grid.states and ACTIONS
        Use stoch_action(a) and attempt_move(s,a)
        """
        new_values = dict()
        new_policy = dict()

        '''from tutorial 7 solution
        
        '''
        for s in self.values:
            # Keep track of maximum value
            action_values = dict()
            max_value = (-np.inf,None)
            if s == self.exit_state:
                total = 0
                new_values[s] = 0
                new_policy[s] = 'wr'
            else:
                new_values[s] = 0
                new_policy[s] = 'wr'
                total = -1200
                for a in self.actions:
                    total = 0

                    # use transition function
                    transition = transition_function(self.environment, s, a)

                    for stateprime in transition:
                        if stateprime not in self.values:
                            x = 2
                        reward = transition[stateprime][0]

                        probability = transition[stateprime][1]
                        fun = reward + (.999999 * self.values[stateprime])
                        great = total
                        great += probability * fun

                        total += probability * (reward + (.9 * self.values[stateprime]))
                        action, t, thi, bi = (a, total,stateprime, self.values[stateprime])
                        action_values[a] = total
                    if total > max_value[0]:
                        max_value = (total, a)

                #new_values[s] = max_value[0]
                #new_policy[s] = max_value[1]
                        # self.values = new_values
                        # self.policy = new_policy



            # Update state value with best action

                new_values[s] = max(action_values.values())
                new_policy[s] = dict_argmax(action_values)




        # Check convergence
        differences = [abs(self.values[s] - new_values[s]) for s in self.values]
        if max(differences) < self.epsilon:
            self.converged = True

        # Update values
        self.values = new_values
        self.policy = new_policy

        # Update values
        self.values = new_values
        self.policy = new_policy

    def print_values(self, policy=False):
        states = sorted([s for s in self.values], key=lambda x: (x.gem_status, x.row, x.col))
        tmp_grid = []
        i = 0

        for r in states:
            row = [(self.values[s], s.gem_status) for s in states if s.row == i]
            i += 1
            tmp_grid.append(row)
            if i > self.environment.n_rows:
                break
        for i, n in enumerate(tmp_grid):

            print('         '.join(["{0:.3f} {1}".format(j, s) for (j, s) in n]))

        print("Converged:", self.converged)


# TODO need to adjust this class for the assignemnt task
class PolicyIteration:  # taken from Tutorial will need to adjust!
    def __init__(self, grid, USE_LIN_ALG=True):
        # self.grid = grid
        self.values = {state: 0 for state in self.grid.states}
        # self.policy = {state: RIGHT for state in self.grid.states}
        self.converged = False
        self.epsilon = 0.001
        self.USE_LIN_ALG = USE_LIN_ALG

    def _lin_alg_iteration(self):
        # use linear algebra for policy evaluation
        # V^pi = R + gamma T^pi V^pi
        # (I - gamma * T^pi) V^pi = R
        # Ax = b; A = (I - gamma * T^pi),  b = R
        pass

    def _policy_iteration(self):
        pass

    def next_iteration(self):
        new_policy = dict()
        # policy evaluation
        if not self.USE_LIN_ALG:
            self._policy_iteration()
        else:
            self._lin_alg_iteration()

    def print_values_and_policy(self):
        for state in self.grid.states:
            pass
            # print(state, get_action_name(self.policy[state]), self.values[state])
        print("Converged:", self.converged)


def run_policy_iteration(max_iter=50):
    # grid = Grid
    pi = PolicyIteration(grid)

    start = time.time()
    print("Initial policy and values:")
    #pi.print_values_and_policy()
    print()

    for i in range(max_iter):
        # pi.next_iteration()
        print("Policy and values after iteration", i + 1)
        # pi.print_values_and_policy()
        print()
        # if pi.converged:
        # break

    end = time.time()
    print("Time to complete", i + 1, "PI iterations")
    print(end - start)


def value_iteration(State, Action, Transition, Reward):
    Value = {s: 0 for s in State}

    while True:
        oldValue = Value.copy()

        for s in State:
            Q = {}
            for a in Action:
                Q[a] = Reward(s, a) + sum(Transition(s_next, s, a) * oldValue[s_next]
                                          for s_next in State)
                Value[s] = max(Q.values())

            if all(oldValue[s] == Value[s] for s in State):
                break

        return Value


# Tut 7
def dict_argmax(d):
    max_value = max(d.values())
    for k, v in d.items():
        if v == max_value:
            return k


class Solver:

    def __init__(self, game_env):
        """
        Constructor for your solver class.

        Any additional instance variables you require can be initialised here.

        Computationally expensive operations should not be included in the constructor, and should be placed in the
        plan_offline() method instead.

        This method has an allowed run time of 1 second, and will be terminated by the simulator if not completed within
        the limit.
        """
        self.game_env = game_env

        #
        #
        # TODO: Initialise any instance variables you require here.
        #
        #

    def plan_offline(self):
        """
        This method will be called once at the beginning of each episode.

        You can use this method to perform value iteration and/or policy iteration and store the computed policy, or
        (optionally) to perform pre-processing for MCTS.

        This planning should not depend on the initial state, as during simulation this is not guaranteed to match the
        initial position listed in the input file (i.e. you may be given a different position to the initial position
        when select_action is called).

        The allowed run time for this method is given by 'game_env.offline_time'. The method will be terminated by the
        simulator if it does not complete within this limit - you should design your algorithm to ensure this method
        exits before the time limit is exceeded.
        """
        t0 = time.time()

        #
        #
        # TODO: Code for offline planning can go here
        #
        #
        states = run_value_iteration(ENVIRONMENT=self.game_env)
        return states
        # optional: loop for ensuring your code exits before the time limit
        while time.time() - t0 < self.game_env.offline_time:
            #
            #
            # TODO: Code for offline planning can go here
            #
            #
            pass

    def select_action(self, state):
        """
        This method will be called each time the agent is called upon to decide which action to perform (once for each
        step of the episode).

        You can use this to retrieve the optimal action for the current state from a stored offline policy (e.g. from
        value iteration or policy iteration), or to perform MCTS simulations from the current state.

        The allowed run time for this method is given by 'game_env.online_time'. The method will be terminated by the
        simulator if it does not complete within this limit - you should design your algorithm to ensure this method
        exits before the time limit is exceeded.

        :param state: the current state, a GameState instance
        :return: action, the selected action to be performed for the current state
        """
        t0 = time.time()
        y = state
        #
        #
        # TODO: Code for retrieving an action from an offline policy, or for online planning can go here
        #
        #
        # self.plan_offline()
        print('hello')
        print('hello', self.plan_offline().values)
        print(self.plan_offline().policy)
        # optional: loop for ensuring your code exits before the time limit

        for i in self.plan_offline().policy:
            if state == i:
                return self.plan_offline().policy[i]

        while time.time() - t0 < self.game_env.online_time:
            #
            #
            # TODO: Code for retrieving an action from an offline policy, or for online planning can go here
            #
            #
            pass

    #
    #
    # TODO: Code for any additional methods you need can go here
    #
    #


def transition_function(environment, state, action):
    """
        Perform the given action on the given state, sample an outcome, and return whether the action was valid, and if
        so, the received reward, the resulting new state and whether the new state is terminal.
        :param state: current GameState
        :param action: an element of self.ACTIONS
        :param seed: random number generator seed (for consistent outcomes between runs)
        :return: (action_is_valid [True/False], received_reward [float], next_state [GameState],
                    state_is_terminal [True/False])
        """
    # state = row , column, gemstatus
    probability = {}

    reward = -1 * environment.ACTION_COST[action]

    # check if the given action is valid for the given state
    if action in {environment.WALK_LEFT, environment.WALK_RIGHT, environment.JUMP}:
        # check walkable ground prerequisite if action is walk or jump
        if environment.grid_data[state.row + 1][state.col] not in environment.WALK_JUMP_ALLOWED_TILES:
            # prerequisite not satisfied
            return probability
    else:
        # check permeable ground prerequisite if action is glide or drop

        if environment.grid_data[state.row + 1][state.col] not in environment.GLIDE_DROP_ALLOWED_TILES:
            # prerequisite not satisfied
            return probability

    # handle each action type separately
    '''Handle for Walk'''
    if action in environment.WALK_ACTIONS:
        if environment.grid_data[state.row + 1][state.col] == environment.SUPER_CHARGE_TILE:
            # sample a random move distance
            '''This is my loop to check walk'''
            for move_dist in environment.super_charge_probs:
                possibility = environment.super_charge_probs[move_dist]

                # set movement direction
                if action == environment.WALK_LEFT:
                    move_dir = -1
                else:
                    move_dir = 1

                next_row, next_col = state.row, state.col
                next_gem_status = state.gem_status

                # move up to the last adjoining supercharge tile
                while environment.grid_data[next_row + 1][next_col + move_dir] == environment.SUPER_CHARGE_TILE:
                    next_col += move_dir
                    # check for collision or game over
                    next_row, next_col, reward, collision, is_terminal = \
                        environment.check_collision_or_terminal(next_row, next_col, reward,
                                                                row_move_dir=0, col_move_dir=move_dir)
                    if collision or is_terminal:
                        break

                # move sampled move distance beyond the last adjoining supercharge tile
                for d in range(move_dist):
                    next_col += move_dir
                    # check for collision or game over
                    next_row, next_col, reward, collision, is_terminal = \
                        environment._check_collision_or_terminal(next_row, next_col, reward,
                                                                 row_move_dir=0, col_move_dir=move_dir)
                    if collision or is_terminal:
                        break

                # check if a gem is collected or goal is reached (only do this for final position of charge)
                next_gem_status, is_terminal = environment._check_gem_collected_or_goal_reached(next_row, next_col,
                                                                                                next_gem_status)

                probability[GameState(next_row, next_col, next_gem_status)] = reward, possibility

            return probability
        else:

            # not on ladder or no fall - set movement direction based on chosen action
            if action == environment.WALK_LEFT:

                col_move_dir = -1
                row_move_dir = 0
                next_row, next_col = (state.row, state.col + col_move_dir)
            else:

                col_move_dir = 1
                row_move_dir = 0
                next_row, next_col = (state.row, state.col + col_move_dir)
            next_gem_status = state.gem_status
            # check for collision or game over
            next_row, next_col, reward, collision, is_terminal = \
                environment._check_collision_or_terminal(next_row, next_col, reward,
                                                         row_move_dir=row_move_dir, col_move_dir=col_move_dir)
            # check if a gem is collected or goal is reached
            next_gem_status, is_terminal = environment._check_gem_collected_or_goal_reached(next_row, next_col,
                                                                                            next_gem_status)

            probability[GameState(next_row, next_col, next_gem_status)] = reward, 1
            # if on ladder, sample whether fall occurs

            if environment.grid_data[state.row + 1][state.col] == environment.LADDER_TILE and \
                    environment.grid_data[state.row + 2][state.col] not in environment.COLLISION_TILES:
                next_row, next_col = state.row + 2, state.col
                # ladder drop
                probability[GameState(next_row, next_col, next_gem_status)] = reward, environment.ladder_fall_prob
                # no drop
                #row ,col = state.row, state.col + col_move_dir
                #gem = state.gem_status

                #probability[GameState(row,col, gem)] = reward, 1 - environment.ladder_fall_prob
            else:
                probability[GameState(next_row, next_col, next_gem_status)] = reward,1
            return probability

    elif action == environment.JUMP:
        if environment.grid_data[state.row + 1][state.col] == environment.SUPER_JUMP_TILE:

            for move_dist in environment.super_jump_probs:
                possibility = environment.super_charge_probs[move_dist]
                next_row, next_col = state.row, state.col
                next_gem_status = state.gem_status

                # move sampled distance upwards
                for d in range(move_dist):
                    next_row -= 1
                    # check for collision or game over
                    next_row, next_col, reward, collision, is_terminal = \
                        environment._check_collision_or_terminal(next_row, next_col, reward, row_move_dir=-1,
                                                                 col_move_dir=0)
                    if collision or is_terminal:
                        break

                # check if a gem is collected or goal is reached (only do this for final position of charge)
                next_gem_status, is_terminal = environment._check_gem_collected_or_goal_reached(next_row,
                                                                                                next_col,
                                                                                                next_gem_status)

                probability[GameState(next_row, next_col, next_gem_status)] = reward, possibility
            return probability

        else:
            next_row, next_col = state.row - 1, state.col
            next_gem_status = state.gem_status
            # check for collision or game over
            next_row, next_col, reward, collision, is_terminal = \
                environment._check_collision_or_terminal(next_row, next_col, reward, row_move_dir=-1, col_move_dir=0)
            # check if a gem is collected or goal is reached
            next_gem_status, is_terminal = environment._check_gem_collected_or_goal_reached(next_row,
                                                                                            next_col,
                                                                                            next_gem_status)
            '''jump not super'''

            probability[GameState(next_row, next_col, next_gem_status)] = reward, 1

        return probability

    elif action in environment.GLIDE_ACTIONS:
        # select probabilities to sample move distance
        if action in {environment.GLIDE_LEFT_1, environment.GLIDE_RIGHT_1}:
            probs = environment.glide1_probs
        elif action in {environment.GLIDE_LEFT_2, environment.GLIDE_RIGHT_2}:
            probs = environment.glide2_probs
        else:
            probs = environment.glide3_probs
        # sample a random move distance
        '''After starting the glide path we will find the move possibility'''
        for move_dist in probs:
            possibility = probs[move_dist]
            # set movement direction
            if action in {environment.GLIDE_LEFT_1, environment.GLIDE_LEFT_2, environment.GLIDE_LEFT_3}:
                move_dir = -1
            else:
                move_dir = 1

            # move sampled distance in chosen direction
            next_row, next_col = state.row + 1, state.col
            next_gem_status = state.gem_status
            if move_dist == 0:
                next_col += move_dir

                # check for collision or game over
                next_row, next_col, reward, collision, is_terminal = \
                    environment.check_collision_or_terminal_glide(next_row, next_col, reward,
                                                                  row_move_dir=0, col_move_dir=move_dir)
                if collision or is_terminal:
                    break

                # check if a gem is collected or goal is reached (only do this for final position of charge)
                next_gem_status, is_terminal = environment._check_gem_collected_or_goal_reached(next_row,
                                                                                                next_col,
                                                                                                next_gem_status)
            else:
                for d in range(move_dist):
                    next_col += move_dir

                    # check for collision or game over
                    next_row, next_col, reward, collision, is_terminal = \
                        environment.check_collision_or_terminal_glide(next_row, next_col, reward,
                                                                      row_move_dir=0, col_move_dir=move_dir)
                    if collision or is_terminal:
                        break

            # check if a gem is collected or goal is reached (only do this for final position of charge)
            next_gem_status, is_terminal = environment._check_gem_collected_or_goal_reached(next_row,
                                                                                            next_col,
                                                                                            next_gem_status)
            if not collision:
                probability[GameState(next_row, next_col, next_gem_status)] = (reward, possibility)
            else:

                next_row, next_col, next_gem_status = state.row, state.col, state.gem_status
                probability[GameState(next_row, next_col, next_gem_status)] = (reward, possibility)
        return probability

    elif action in environment.DROP_ACTIONS:
        move_dist = {environment.DROP_1: 1, environment.DROP_2: 2, environment.DROP_3: 3}[action]

        # drop by chosen distance
        next_row, next_col = state.row, state.col
        next_gem_status = state.gem_status
        for d in range(move_dist):
            next_row += 1

            # check for collision or game over
            next_row, next_col, reward, collision, is_terminal = \
                environment.check_collision_or_terminal_glide(next_row, next_col, reward, row_move_dir=1,
                                                              col_move_dir=0)
            if collision or is_terminal:
                break

        # check if a gem is collected or goal is reached (only do this for final position of charge)
        next_gem_status, is_terminal = environment._check_gem_collected_or_goal_reached(next_row,
                                                                                        next_col,
                                                                                        next_gem_status)

        probability[GameState(next_row, next_col, next_gem_status)] = (reward, 1)
        return probability

    else:
        assert False, '!!! Invalid action given to perform_action() !!!'


def run_value_iteration(ENVIRONMENT, max_iter=50):
    #ENVIRONMENT = game_env.GameEnv(r"testcases/a2-t1.txt")
    grid = ValueIteration(ENVIRONMENT.grid_data, ENVIRONMENT)
    vi = ValueIteration(grid, ENVIRONMENT)

    start = time.time()
    # print("Initial values:)")
    #vi.print_values()

    #print()

    for i in range(max_iter):
        vi.next_iteration()
        print("Values after iteration", i + 1)
        vi.print_values(policy=True)
        #print()
        if vi.converged:
            break

    end = time.time()
    #print(vi.policy)
    print("Time to complete", i + 1, "VI iterations")
    print(end - start)
    return vi


if __name__ == "__main__":
    #run_value_iteration(game_env.GameEnv(r"testcases/a2-t1.txt"))

    solver = Solver(game_env.GameEnv(r"testcases/a2-t1.txt"))

    init_state = solver.game_env.get_init_state()
    test_state = GameState(1, 1, (0,))
    for i in game_env.GameEnv.ACTIONS:
        t = transition_function(game_env.GameEnv(r"testcases/a2-t1.txt"), test_state, i)
    # VI = ValueIteration(r"testcases/a2-t4.txt")
        #print(i,t)
    '''
    for a in game_env.GameEnv.ACTIONS:
        thing = transition_function(solver.game_env, init_state, a)
        for i in thing:
            print(f"{a} {i} {thing[i]}")
    '''
    # TODO check why glide is being changed to a lower number than it should be
