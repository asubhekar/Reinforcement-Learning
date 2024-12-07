import torch
import random
import numpy as np 
from collections import deque
from snakeai import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_game = 0
        self.epsilon = 0 # Controlling the randomness
        self.gamma = 0.9 # Discount rate 
        self.memory = deque(maxlen = MAX_MEMORY) # Poplefft

        # model, trainer
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        pass 

    def get_state(self, game):
        # Getting the head from the game
        head = game.snake[0]

        # Taking the points around the head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # Getting Current Direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
                # Danger straight
                (dir_r and game.is_collision(point_r))or
                (dir_l and game.is_collision(point_l))or
                (dir_u and game.is_collision(point_u))or
                (dir_d and game.is_collision(point_d)),

                # Danger right
                (dir_r and game.is_collision(point_d))or
                (dir_l and game.is_collision(point_u))or
                (dir_u and game.is_collision(point_r))or
                (dir_d and game.is_collision(point_l)),

                # Danger Left
                (dir_r and game.is_collision(point_u))or
                (dir_l and game.is_collision(point_d))or
                (dir_u and game.is_collision(point_l))or
                (dir_d and game.is_collision(point_r)),

                # Movement direction 
                dir_l,
                dir_r,
                dir_u,
                dir_d,

                # Food Location
                game.food.x < game.head.x,
                game.food.x > game.head.x,
                game.food.y < game.head.y,
                game.food.y > game.head.y,
        ]
        # Setting the datatype of the return variable to int to convert boolean flags into numbers
        return np.array(state, dtype = int)

    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # Automatically pops left if the append is after the MAX_MEMORY
       

    def train_long_memory(self):
        ''' We will batch the data into a tensor based on batch size.
            If the memory does not hold enough information to statisfy the batch size we take all the available samples. 
            If the memory has more elements than the batch size, then we randomly pick samples based on the batch size.
        '''
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # returns list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

        

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        
    
    def get_action(self, state):
        '''Starting with random moves: trade off exploration / exploitation
        We will use the self.epsilon to set as a hyper parameter
        '''
        self.epsilon = 80 - self.n_game
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            '''We pass a state to the model and get predictions on the next action'''
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move
        

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # Getting old state
        state_old = agent.get_state(game)

        # Getting the next move
        final_move = agent.get_action(state_old)

        # Performing the move in game
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Training for short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember the actions and state
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Training the old memory (experience replay)
            game.reset()
            agent.n_game += 1

            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_game, 'Score', score, 'Record', record)

            plot_scores.append(score)

            total_score += score
            mean_score = total_score / agent.n_game
            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores) 
        


if __name__== "__main__":
    try:
    # Main execution block
        train()
    except Exception as e:
        print(f"Error occurred: {e}")
        