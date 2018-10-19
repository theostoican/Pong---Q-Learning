# Theodor Stoican, 2018

# General imports
from copy import copy
from random import choice, random
from argparse import ArgumentParser
from time import sleep
import sys
import numpy as np
import pygame
from pygame.locals import *
import pong
import pprint

pp = pprint.PrettyPrinter(indent=2)

AGENT_STRATEGIES = ["GREEDY", "RANDOM", "E-GREEDY"]
ENEMY_STRATEGIES = ["RANDOM", "GREEDY", "PERFECT"]

class QLearning:
    def __init__(self, agent_str, enemy_str):
        self.game = pong.Game()
        self.agent_str = agent_str
        self.enemy_str = enemy_str

    # Random strategy
    def random_act(self, legal_actions):
        return choice(legal_actions)

    # greedy strategy
    def greedy(self, Q, state, legal_actions):
        max_val = None
        opt_act = None

        for act in legal_actions:
            if act not in Q[state]:
                Q[state][act] = 0
            if max_val is None or max_val < Q[state][act]:
                max_val = Q[state][act]
                opt_act = act
        return opt_act

    # epsilon-greedy strategy
    def epsilon_greedy(self, Q, state, legal_actions, epsilon):
        not_visited = []

        for act in legal_actions:
            if act not in Q[state]:
                not_visited.append(act)

        # Randomly choose from those which are yet to be visited
        if len(not_visited) > 0:
            return choice(not_visited)

        # Choose with epsilon probability smth, and with 1-epsilon smth else
        if np.random.uniform(0,1) < epsilon:
            return choice(legal_actions)
        else:
            max_val = None
            opt_act = None
            for act in legal_actions:
                if max_val is None or max_val < Q[state][act]:
                    max_val = Q[state][act]
                    opt_act = act
            return opt_act

    # A greedy strategy for the enemy; need to make a translation of the ball
    # direction and of the racket (to be able to use the utilities that were
    # learned for the agent)
    def enemy_greedy(self, Q, state, legal_actions):
        enemy_pos = state[2]
        ball_pos = state[0]
        ball_vel = state[1]
        max_val = None
        opt_act = None

        # Make the translation
        enemy_pos = (pong.HALF_PAD_WIDTH, enemy_pos[1])
        ball_pos = (pong.WIDTH - ball_pos[0], ball_pos[1])
        ball_vel = (-ball_vel[0], ball_vel[1])

        # Return randomly if this state was not previously searched
        state = (ball_pos, ball_vel, enemy_pos)
        if state not in Q:
            return choice(legal_actions)

        # Choose greedily the best action
        for act in legal_actions:
            if act not in Q[state]:
                Q[state][act] = 0
            if max_val is None or max_val < Q[state][act]:
                max_val = Q[state][act]
                opt_act = act
        return opt_act

    # A perfect strategy for the enemy. Moves its racket after the ball
    def perfect(self, state, legal_actions):
        enemy_pos = state[2]
        ball_pos = self.game.predict_next_ball_pos(state)
        if np.random.uniform(0,1) < pong.ALPHA:
            if enemy_pos[1] < ball_pos[1]:
                # Go up
                return pong.ACTIONS[0]
            elif enemy_pos[1] > ball_pos[1]:
                # Go down
                return pong.ACTIONS[1]
            else:
                # Stay
                return pong.ACTIONS[2]
        else:
            return choice(legal_actions)


    def best_action(self, Q, state, legal_actions):
        max_val = None
        opt_act = None
        if state not in Q:
            return choice(legal_actions)
        for act in legal_actions:
            if act in Q[state]:
                if max_val is None or max_val < Q[state][act]:
                    max_val = Q[state][act]
                    opt_act = act
        return opt_act

    # Run the game with the results that we have gathered so far
    def playout(self, Q, args):
        curr_game = pong.Game()
        state = curr_game.get_initial_state()
        score = 0

        while True:
            if ql.game.is_final_state(state, score):
                break
            state, reward = ql.simulate(Q, state, ql.game, args)
            score += reward
        return score

    # Getter for agent action, according to its strategy
    def get_agent_action(self, Q, look_up_state, actions, epsilon):
        agent_action = None

        if self.agent_str == "E-GREEDY":
            agent_action = self.epsilon_greedy(Q, look_up_state, actions, args.epsilon)
        elif self.agent_str == "GREEDY":
            agent_action = self.greedy(Q, look_up_state, actions)
        elif self.agent_str == "RANDOM":
            agent_action = self.random_act(actions)

        return agent_action

    # Getter for enemy action, according to its strategy
    def get_enemy_action(self, Q, enemy_look_up_state, actions):
        enemy_action = None

        if self.enemy_str == "RANDOM":
            enemy_action = self.random_act(actions)
        elif self.enemy_str == "GREEDY":
            enemy_action = self.enemy_greedy(Q, enemy_look_up_state, actions)
        elif self.enemy_str == "PERFECT":
            enemy_action = self.perfect(enemy_look_up_state, actions)

        return enemy_action

    # The main Q-learning routine
    def q_learning(self, args):
        Q = {}
        train_scores = []
        eval_scores = []
        alpha = args.learning_rate
        discount = args.discount

        # for each episode ...
        for train_ep in range(1, args.train_episodes + 1):

            # Restart the game
            self.game = pong.Game()
            # ... get the initial state,
            score = 0
            state = self.game.get_initial_state()
            # display current state and sleep
            if args.verbose:
                self.display_state(state); sleep(args.sleep)

            # while current state is not terminal
            while not self.game.is_final_state(state, score):

                # choose one of the legal actions
                actions = self.game.get_legal_actions()
                #print(state)
                # Do not take into account the enemy's state
                look_up_state = (state[0], state[1], state[2])
                if look_up_state not in Q:
                    Q[look_up_state] = {}

                enemy_look_up_state = (state[0], state[1], state[3])

                # Choose the actions for the agent and the enemy according to the
                # desired strategies
                agent_action = self.get_agent_action(Q, look_up_state, actions, args.epsilon)
                enemy_action = self.get_enemy_action(Q, enemy_look_up_state, actions)

                # apply action and get the next state and the reward
                _state, reward = self.game.apply_action(agent_action, enemy_action, state)
                score += reward

                # Initialize Q for the current state
                if agent_action not in Q[look_up_state]:
                    Q[look_up_state][agent_action] = 0

                # get the next state valid actions and initialize if they are not already
                # present in Q
                next_state_act = self.game.get_legal_actions()
                look_up_next_state = (_state[0], _state[1], _state[2])
                if look_up_next_state not in Q:
                    Q[look_up_next_state] = {}
                for act in next_state_act:
                    if act not in Q[look_up_next_state]:
                        Q[look_up_next_state][act] = 0

                max_pair = None
                for act in next_state_act:
                    if max_pair is None:
                        max_pair = Q[look_up_next_state][act]
                    elif max_pair < Q[look_up_next_state][act]:
                        max_pair = Q[look_up_next_state][act]

                Q[look_up_state][agent_action] = Q[look_up_state][agent_action] + alpha * (reward + discount * max_pair - Q[look_up_state][agent_action])
                if args.verbose:
                    print(msg); display_state(state); sleep(args.sleep)
                state = _state

            train_scores.append(score)

            # evaluate the policy
            if train_ep % args.eval_every == 0:
                avg_score = .0

                # test for some playouts
                for i in range(0, args.eval_episodes):
                    avg_score += self.playout(Q, args)
                avg_score /= args.eval_episodes

                eval_scores.append(avg_score)


        # --------------------------------------------------------------------------

        if args.final_show:
            window = pygame.display.set_mode((pong.WIDTH, pong.HEIGHT), 0, 32)
            ql.game = pong.Game()
            state = ql.game.get_initial_state()
            score = 0
            while True:
                fps.tick(2)
                if ql.game.is_final_state(state, score):
                    ql.draw(window, Q, state, ql.game, args)
                    break
                state, reward = ql.draw(window, Q, state, ql.game, args)
                score += reward
                pygame.display.update()


        if args.plot_scores:
            from matplotlib import pyplot as plt
            import numpy as np
            plt.xlabel("Episode")
            plt.ylabel("Average score")
            plt.plot(
                np.linspace(1, args.train_episodes, args.train_episodes),
                np.convolve(train_scores, [0.2,0.2,0.2,0.2,0.2], "same"),
                linewidth = 1.0, color = "blue"
            )
            plt.plot(
                np.linspace(args.eval_every, args.train_episodes, len(eval_scores)),
                eval_scores, linewidth = 2.0, color = "red"
            )
            plt.show()
        return Q

    # Simulate the game without showing the movements of the rackets and the ball
    def simulate(self, Q, state, game, args):
        actions = game.get_legal_actions()
        look_up_state = (state[0], state[1], state[2])

        # Choose action for agent
        agent_action = self.best_action(Q, look_up_state, actions)

        # Choose action for enemy
        enemy_look_up_state = (state[0], state[1], state[3])
        enemy_action = self.get_enemy_action(Q, enemy_look_up_state, actions)

        _state, reward = self.game.apply_action(agent_action, enemy_action, state)

        ball_pos = state[0]
        agent_racket_pos = state[2]
        enemy_racket_pos = state[3]

        return _state, reward

    # Simulate the game, while also showing the movements in the GUI
    def draw(self, canvas, Q, state, game, args):
        # Prepare the table
        canvas.fill(pong.BLACK)
        pygame.draw.line(canvas, pong.WHITE, [pong.WIDTH // 2, 0],[pong.WIDTH // 2, pong.HEIGHT], 1)
        pygame.draw.line(canvas, pong.WHITE, [pong.PAD_WIDTH, 0],[pong.PAD_WIDTH, pong.HEIGHT], 1)
        pygame.draw.line(canvas, pong.WHITE, [pong.WIDTH - pong.PAD_WIDTH, 0],[pong.WIDTH - pong.PAD_WIDTH, pong.HEIGHT], 1)
        pygame.draw.circle(canvas, pong.WHITE, [pong.WIDTH//2, pong.HEIGHT//2], 70, 1)

        actions = game.get_legal_actions()
        look_up_state = (state[0], state[1], state[2])

        # Choose action for agent
        agent_action = self.best_action(Q, look_up_state, actions)

        # Choose action for enemy
        enemy_look_up_state = (state[0], state[1], state[3])
        enemy_action = self.get_enemy_action(Q, enemy_look_up_state, actions)

        _state, reward = self.game.apply_action(agent_action, enemy_action, state)

        ball_pos = state[0]
        agent_racket_pos = state[2]
        enemy_racket_pos = state[3]


        # Draw rackets and ball
        pygame.draw.circle(canvas, pong.RED, ball_pos, pong.BALL_RADIUS, 0)
        pygame.draw.polygon(canvas, pong.GREEN,
            [[agent_racket_pos[0] - pong.HALF_PAD_WIDTH, agent_racket_pos[1] - pong.HALF_PAD_HEIGHT],
            [agent_racket_pos[0] - pong.HALF_PAD_WIDTH, agent_racket_pos[1] + pong.HALF_PAD_HEIGHT],
            [agent_racket_pos[0] + pong.HALF_PAD_WIDTH, agent_racket_pos[1] + pong.HALF_PAD_HEIGHT],
            [agent_racket_pos[0] + pong.HALF_PAD_WIDTH, agent_racket_pos[1] - pong.HALF_PAD_HEIGHT]], 0)
        pygame.draw.polygon(canvas, pong.GREEN,
            [[enemy_racket_pos[0] - pong.HALF_PAD_WIDTH, enemy_racket_pos[1] - pong.HALF_PAD_HEIGHT],
            [enemy_racket_pos[0] - pong.HALF_PAD_WIDTH, enemy_racket_pos[1] + pong.HALF_PAD_HEIGHT],
            [enemy_racket_pos[0] + pong.HALF_PAD_WIDTH, enemy_racket_pos[1] + pong.HALF_PAD_HEIGHT],
            [enemy_racket_pos[0] + pong.HALF_PAD_WIDTH, enemy_racket_pos[1] - pong.HALF_PAD_HEIGHT]], 0)


        return _state, reward


if __name__ == "__main__":
    pygame.init()
    fps = pygame.time.Clock()
    # Set up arguments for the Q-learning algorithm
    parser = ArgumentParser()

    # Meta-parameters
    parser.add_argument("--agent_strategy", default = "RANDOM", help = "Agent strategy");
    parser.add_argument("--enemy_strategy", default = "RANDOM", help = "Enemy strategy");
    parser.add_argument("--learning_rate", type = float, default = 0.2,
                        help = "Learning rate")
    parser.add_argument("--discount", type = float, default = 0.999999999999,
                        help = "Value for the discount factor")
    parser.add_argument("--epsilon", type = float, default = 0.02,
                        help = "Probability to choose a random action.")
    # Training and evaluation episodes
    parser.add_argument("--train_episodes", type = int, default = 5000,
                        help = "Number of episodes")
    parser.add_argument("--eval_every", type = int, default = 100,
                        help = "Evaluate policy every ... games.")
    parser.add_argument("--eval_episodes", type = float, default = 10,
                        help = "Number of games to play for evaluation.")
    # Display
    parser.add_argument("--verbose", dest="verbose",
                        action = "store_true", help = "Print each state")
    parser.add_argument("--plot", dest="plot_scores", action="store_true",
                        help = "Plot scores in the end")
    parser.add_argument("--sleep", type = float, default = 0.1,
                        help = "Seconds to 'sleep' between moves.")
    parser.add_argument("--final_show", dest = "final_show",
                        action = "store_true",
                        help = "Demonstrate final strategy.")
    args = parser.parse_args()

    print(args.agent_strategy, args.enemy_strategy)
    ql = QLearning(args.agent_strategy, args.enemy_strategy)

    # Learning phase
    ql.q_learning(args)
