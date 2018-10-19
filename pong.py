import random
import pygame, sys
from pygame.locals import *
from copy import copy

# colors
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLACK = (0,0,0)

# actions
ACTIONS = ["UP", "DOWN", "STAY"]
ACTION_EFFECTS = {
    "UP": 100,
    "DOWN": -100,
    "STAY": 0
}

# rewards
MOVE_REWARD = -0.1
WIN_REWARD = 40
LOSE_REWARD = -40

# globals
WIDTH = 1100
HEIGHT = 700
BALL_RADIUS = 50
PAD_WIDTH = 100
PAD_HEIGHT = 100
HALF_PAD_WIDTH = PAD_WIDTH // 2
HALF_PAD_HEIGHT = PAD_HEIGHT // 2
MAX_ITER = 5
BALL_SPEED = 100
ALPHA = 0.3

class Ball:
    def __init__(self):
        possible_vel_x = [BALL_SPEED, 0, -BALL_SPEED, -BALL_SPEED, 0, BALL_SPEED]
        possible_vel_y = [BALL_SPEED, BALL_SPEED, BALL_SPEED, -BALL_SPEED, -BALL_SPEED, -BALL_SPEED]

        # The ball starts from the middle
        self.ball_pos = (WIDTH//2, HEIGHT//2)
        # Randomly choose the direction
        rand_pos = random.randint(0, len(possible_vel_y) - 1)
        self.ball_vel = (possible_vel_y[rand_pos], possible_vel_x[rand_pos])

class Game:
    def __init__(self):
        self.agent_racket_pos = (HALF_PAD_WIDTH, HEIGHT//2)
        self.enemy_racket_pos = (WIDTH - HALF_PAD_WIDTH, HEIGHT//2)
        self.agent_score = 0
        self.enemy_score = 0
        self.agent_racket_vel = 0
        self.enemy_racket_vel = 0
        self.ball = Ball()
        self.total_iter = 0
        self.initial_state = (self.ball.ball_pos, self.ball.ball_vel, self.agent_racket_pos, self.enemy_racket_pos)

    def get_initial_state(self):
        return self.initial_state

    # Return all legal actions for the agent or the enemy
    def get_legal_actions(self):
        return copy(ACTIONS)

    def is_final_state(self, state, score):
        current_ball_pos = state[0]

        if current_ball_pos[0] <= BALL_RADIUS or \
            current_ball_pos[0] >= WIDTH - BALL_RADIUS or \
            score <= -MAX_ITER:
            return True
        return False

    # Predict where the ball will be at t+1, if we know where it is
    # at time t
    def predict_next_ball_pos(self, state):
        current_ball_pos = state[0]
        current_ball_vel = state[1]
        future_ball_pos = current_ball_pos
        future_ball_vel = current_ball_vel

        # Ball collision tests with the top and bottom walls
        if current_ball_pos[1] <= BALL_RADIUS or current_ball_pos[1] >= HEIGHT - BALL_RADIUS:
            future_ball_vel = (future_ball_vel[0], -current_ball_vel[1])


        # Update the coordinates of the ball
        future_ball_pos = (future_ball_pos[0] + future_ball_vel[0],
            future_ball_pos[1] + future_ball_vel[1])

        return future_ball_pos

    # Return the new state after applying a particular action
    def apply_action(self, agent_action, enemy_action, state):
        # One particular racket can go either up, down or stay at the same position
        agent_move = ACTION_EFFECTS[agent_action]
        enemy_move = ACTION_EFFECTS[enemy_action]
        next_state = None
        reward = MOVE_REWARD
        current_ball_pos = state[0]
        current_ball_vel = state[1]
        current_agent_racket = state[2]
        current_enemy_racket = state[3]
        future_ball_pos = current_ball_pos
        future_ball_vel = current_ball_vel

        # Ball collision tests with the top and bottom walls
        if current_ball_pos[1] <= BALL_RADIUS or current_ball_pos[1] >= HEIGHT - BALL_RADIUS:
            future_ball_vel = (future_ball_vel[0], -current_ball_vel[1])

        # Ball collision with rackets
        if (current_ball_pos[0] - BALL_RADIUS <= BALL_RADIUS + HALF_PAD_WIDTH and
            current_ball_pos[1] in range(current_agent_racket[1] - HALF_PAD_HEIGHT, current_agent_racket[1] + HALF_PAD_HEIGHT)) or \
            (current_ball_pos[0]  + BALL_RADIUS >= WIDTH - BALL_RADIUS - HALF_PAD_WIDTH and
                current_ball_pos[1] in range(current_enemy_racket[1] - HALF_PAD_HEIGHT, current_enemy_racket[1] + HALF_PAD_HEIGHT)):
            future_ball_vel = (-current_ball_vel[0], future_ball_vel[1])
        else:
            # Ball collision with left and right walls
            if current_ball_pos[0] - BALL_RADIUS <= PAD_WIDTH:
                reward = LOSE_REWARD
            elif current_ball_pos[0] + BALL_RADIUS >= WIDTH - PAD_WIDTH:
                reward = WIN_REWARD

        # Update the coordinates of the ball
        future_ball_pos = (future_ball_pos[0] + future_ball_vel[0],
            future_ball_pos[1] + future_ball_vel[1])

        # Construct the next possible coordinates of the rackets
        if current_agent_racket[1] < HEIGHT - HALF_PAD_HEIGHT and agent_move > 0 or \
            current_agent_racket[1] > HALF_PAD_HEIGHT and agent_move < 0:
             current_agent_racket = (current_agent_racket[0], current_agent_racket[1] + agent_move)
        # can go up or down
        if current_enemy_racket[1] < HEIGHT - HALF_PAD_HEIGHT and enemy_move > 0 or \
            current_enemy_racket[1] > HALF_PAD_HEIGHT and enemy_move < 0:
             current_enemy_racket = (current_enemy_racket[0], current_enemy_racket[1] + enemy_move)

        next_state = (future_ball_pos, future_ball_vel, current_agent_racket, current_enemy_racket)

        return next_state, reward
