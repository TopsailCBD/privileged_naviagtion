import math
import sys

import numpy as np
from astar import AStar
from scipy.signal import convolve2d


class MazeSolver(AStar):

    """sample use of the astar algorithm. In this exemple we work on a maze made of ascii characters,
    and a 'node' is just a (x,y) tuple that represents a reachable position"""

    def __init__(self, maze):
        self.lines = maze
        self.width = len(self.lines[0])
        self.height = len(self.lines)
        self.offsets = [(-1,-1),(-1,0),(-1,1),
                        (0,-1),        (0,1),
                        (1,-1),(1,0),(1,1)]

    def heuristic_cost_estimate(self, n1, n2):
        """computes the 'direct' distance between two (x,y) tuples"""
        (x1, y1) = n1
        (x2, y2) = n2
        return math.hypot(x2 - x1, y2 - y1)

    def distance_between(self, n1, n2):
        """this method returns cost of moving between two adjacent 'neighbors'"""
        x1, y1 = n1
        x2, y2 = n2
        return math.hypot(x2 - x1, y2 - y1)

    def neighbors(self, node):
        """ for a given coordinate in the maze, returns up to 4 adjacent(north,east,south,west)
            nodes that can be reached (=any adjacent coordinate that is not a wall)
        """
        x, y = node
        neighbor_idx = [(x + dx, y + dy) for (dx, dy) in self.offsets]
        return[(nx, ny) for nx, ny in neighbor_idx if 0 <= nx < self.height and 0 <= ny < self.width and self.lines[nx][ny] == 0]

def solve_maze(m,st,en):
    
    # let's solve it
    foundPath = list(MazeSolver(m).astar(st, en))

    return list(foundPath)

if __name__ == '__main__':
    # generate an array maze
    m = np.array([
     [0,0,0,0,0],
     [0,0,0,0,0],
     [1,1,1,0,0],
     [0,0,0,0,0],
     [0,0,0,0,0],
     [0,0,0,0,1]
    ])

    k1 = np.array([
     [0,1,0],
     [1,1,1],
     [0,1,0]
    ])
    
    k2 = np.array([
        [0,0,1,0,0],
        [0,1,1,1,0],
        [1,1,1,1,1],
        [0,1,1,1,0],
        [0,0,1,0,0]
    ])
    
    m = convolve2d(m,k1,mode='same',boundary='fill',fillvalue=0)
    print(m)
    
 	# what is the size of it?
    w = len(m[0])
    h = len(m)

    start = (0, 0)  # we choose to start at the upper left corner
    goal = (w - 1, 0)  # we want to reach the lower right corner
 
    print(solve_maze(m,start,goal))
