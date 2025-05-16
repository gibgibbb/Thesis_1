import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import distance_transform_edt

# Grid setup (10x10 synthetic example)
grid_size = 10
grid = np.zeros((grid_size, grid_size))  # 0: empty, 1: walkable, 3: exit, 4: fire
grid[1:9, 1:9] = 1  # Walkable area
grid[0, 5] = 3      # Exit
grid[5, 5] = 4      # Initial fire
people = np.zeros_like(grid)  # 1: person present
people[5, 1] = 1     # Initial person location

# Static floor field (distance to exit)
exit_pos = np.where(grid == 3)
static_field = distance_transform_edt(grid != 3)

# Dynamic floor field
dynamic_field = np.zeros_like(grid)
decay_rate = 0.1
diffusion_rate = 0.05

# Fire spread probability
fire_spread_prob = 0.2

def update(frame):
    global grid, people, dynamic_field
    
    # Fire spread
    fire_cells = np.where(grid == 4)
    for x, y in zip(*fire_cells):
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < grid_size and 0 <= new_y < grid_size and 
                grid[new_x, new_y] == 1 and np.random.rand() < fire_spread_prob):
                grid[new_x, new_y] = 4
    
    # Pedestrian movement
    new_people = np.zeros_like(people)
    people_cells = np.where(people == 1)
    for x, y in zip(*people_cells):
        if grid[x, y] == 3:  # Reached exit
            continue
        # Calculate transition probabilities
        neighbors = [(x+dx, y+dy) for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]
                     if 0 <= x+dx < grid_size and 0 <= y+dy < grid_size]
        probs = []
        for nx, ny in neighbors:
            if grid[nx, ny] in [1, 3]:  # Walkable or exit
                prob = static_field[nx, ny] + dynamic_field[nx, ny]
                probs.append((prob, nx, ny))
        if probs:
            _, next_x, next_y = max(probs, key=lambda p: p[0])
            new_people[next_x, next_y] = 1
            dynamic_field[x, y] += 1  # Leave trace
    
    # Update dynamic field
    dynamic_field = (1 - decay_rate) * dynamic_field + diffusion_rate * (
        np.roll(dynamic_field, 1, axis=0) + np.roll(dynamic_field, -1, axis=0) +
        np.roll(dynamic_field, 1, axis=1) + np.roll(dynamic_field, -1, axis=1))
    people = new_people
    
    # Visualization
    ax.clear()
    ax.imshow(grid + people, cmap='hot', interpolation='nearest')
    ax.set_title(f"Time Step: {frame}")
    return ax,

# Animation
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=range(50), interval=1000)
plt.show()