
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()



class Direction(Enum):
    RIGHT = 1
    LEFT = 2 
    DOWN = 4
    UP = 3


Point = namedtuple('Point','x,y')

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
font = pygame.font.Font('arial.ttf', 25)
BLOCK_SIZE = 20
SPEED = 10


class SnakeGameHuman:

    def __init__(self, w = 640, h = 480):
        self.w = w
        self.h = h

        # Initializing the display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()

        # Initializing the game state
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, Point(self.head.x - BLOCK_SIZE, self.head.y), Point(self.head.x - (2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()

    def _move(self, direction):

        x = self.head.x
        y = self.head.y

        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x,y)

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

        self.food = Point(x,y)

        if self.food in self.snake:
            self._place_food()

    def _is_collision(self):
        
        # Checking if the snake hits the boundary
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE  or self.head.y < 0:
            return True
        
        # Checking if the snake hits itself
        if self.head in self.snake[1:]:
            return True

        return False


    
    def play_step(self):

        # Taking user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    self.direction = Direction.LEFT
                elif event.key ==pygame.K_d:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_w:
                    self.direction = Direction.UP
                elif event.key == pygame.K_s:
                    self.direction = Direction.DOWN

        # Character movement
        self._move(self.direction) # Updating the head
        self.snake.insert(0, self.head)

        # Check if the game is over
        game_over = False

        if self._is_collision():
            game_over = True
            return game_over, self.score
        
        # Putting new food if the old one is consumed by the snake or making the snake move ahead
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        
        self._update_ui()
        self.clock.tick(SPEED)
        return game_over, self.score
    

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render('Score = ' +str(self.score), True, WHITE)

        self.display.blit(text, [0,0])
        pygame.display.flip()


if __name__ == '__main__':
    game = SnakeGameHuman()

    while True:
        game_over, score = game.play_step()

        if game_over == True:
            break

    print("Final Score = ", score)

    pygame.quit()
    







