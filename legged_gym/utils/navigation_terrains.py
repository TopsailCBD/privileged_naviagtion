# Define some special terrains for navigation tasks

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import gymutil, gymapi
from math import sqrt

def discrete_obstacles_terrain(terrain, max_height, min_size, max_size, num_rects, platform_size=1.):
    """
    Generate a terrain with gaps

    Parameters:
        terrain (terrain): the terrain
        max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
        min_size (float): minimum size of a rectangle obstacle [meters]
        max_size (float): maximum size of a rectangle obstacle [meters]
        num_rects (int): number of randomly generated obstacles
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    max_height = int(max_height / terrain.vertical_scale)
    min_size = int(min_size / terrain.horizontal_scale)
    max_size = int(max_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    (i, j) = terrain.height_field_raw.shape
    height_range = [-max_height, -max_height // 2, max_height // 2, max_height]
    width_range = range(min_size, max_size, 4)
    length_range = range(min_size, max_size, 4)

    for _ in range(num_rects):
        width = np.random.choice(width_range)
        length = np.random.choice(length_range)
        start_i = np.random.choice(range(0, i-width, 4))
        start_j = np.random.choice(range(0, j-length, 4))
        terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = np.random.choice(height_range)

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain


def box_terrain(terrain, max_height, min_size, max_size, num_rects, wall_height):
    """
    Generate a terrain with walls surrounding a flat platform while obstacles on the platform 
    """
    
    # switch parameters to discrete units
    max_height = int(max_height / terrain.vertical_scale)
    
    wall_height = int(wall_height / terrain.vertical_scale)
    min_size = int(min_size / terrain.horizontal_scale)
    max_size = int(max_size / terrain.horizontal_scale)
    
    num_rects = int(num_rects)

    (i, j) = terrain.height_field_raw.shape
    height_range = range(max_height // 4, max_height, 3)
    width_range = range(min_size, max_size, 4)
    length_range = range(min_size, max_size, 4)

    for _ in range(num_rects):
        width = np.random.choice(width_range)
        length = np.random.choice(length_range)
        start_i = np.random.choice(range(0, i-width, 4))
        start_j = np.random.choice(range(0, j-length, 4))
        terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = np.random.choice(height_range)

    # Generate walls
    terrain.height_field_raw[:2, :] = wall_height
    terrain.height_field_raw[:, :2] = wall_height
    terrain.height_field_raw[-2:, :] = wall_height
    terrain.height_field_raw[:, -2:] = wall_height

    # Generate a door
    
    door_width = max_size // 2
    # door_height = np.random.choice(height_range)
    door_wall_idx = np.random.choice([0, 1, 2, 3])
    if door_wall_idx in [0,2]:
        door_start_j = np.random.choice(range(0, j-door_width, 4))
        if door_wall_idx == 0:
            terrain.height_field_raw[:10,door_start_j:door_start_j+door_width] = 0
        elif door_wall_idx == 2:
            terrain.height_field_raw[-10:,door_start_j:door_start_j+door_width] = 0
    else:
        door_start_i = np.random.choice(range(0, i-door_width, 4))
        if door_wall_idx == 1:
            terrain.height_field_raw[door_start_i:door_start_i+door_width,:10] = 0
        elif door_wall_idx == 3:
            terrain.height_field_raw[door_start_i:door_start_i+door_width,-10:] = 0
        
    return terrain