import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Grid parameters
GRID_SIZE = 50
CELL_SIZE = 10
EXIT_POINTS = [(0, 0), (0, 25), (0, 49), (49, 0), (49, 25), (49, 49)]
FIRE_START = (25, 25)

# States
EMPTY = 0
PERSON = 1
FIRE = 2
EXIT = 3
EVACUATED = 4

# Initialize grid
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
grid[FIRE_START] = FIRE
for exit_pos in EXIT_POINTS:
    grid[exit_pos] = EXIT

# Populate grid with people
np.random.seed(42)
person_positions = np.random.choice(GRID_SIZE * GRID_SIZE, size=100, replace=False)
for pos in person_positions:
    x, y = divmod(pos, GRID_SIZE)
    if grid[x, y] == EMPTY:
        grid[x, y] = PERSON

# Fire spread parameters
P_BURN = 0.58
WIND_FACTOR = 0.045

# Evacuation parameters
MOVE_PROB = 0.8

# Precompute distance to nearest exit
distance_to_exit = np.full((GRID_SIZE, GRID_SIZE), np.inf)
for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        distances = [abs(x - ex) + abs(y - ey) for ex, ey in EXIT_POINTS]
        distance_to_exit[x, y] = min(distances)

def get_neighbors(grid, x, y, radius=1):
    """Get neighboring cells within radius."""
    neighbors = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if i == 0 and j == 0:
                continue
            nx, ny = x + i, y + j
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                neighbors.append((nx, ny))
    return neighbors

def is_safe_from_fire(grid, x, y):
    """Return True if cell (x, y) is not adjacent to FIRE."""
    for nx, ny in get_neighbors(grid, x, y):
        if grid[nx, ny] == FIRE:
            return False
    return True

def update_fire(grid):
    """Update fire spread based on probabilistic rules."""
    new_grid = grid.copy()
    fire_cells = np.where(grid == FIRE)
    for x, y in zip(fire_cells[0], fire_cells[1]):
        for nx, ny in get_neighbors(grid, x, y):
            if grid[nx, ny] in [EMPTY, PERSON] and np.random.random() < P_BURN:
                new_grid[nx, ny] = FIRE
    return new_grid

def update_people(grid):
    """Update people movement with swarm-based routing."""
    new_grid = grid.copy()
    person_cells = list(zip(*np.where(grid == PERSON)))
    np.random.shuffle(person_cells)
    
    for x, y in person_cells:
        if np.random.random() < MOVE_PROB:
            possible_moves = []
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            current_dist = distance_to_exit[x, y]
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and
                        (grid[nx, ny] == EMPTY or grid[nx, ny] == EXIT) and
                        is_safe_from_fire(grid, nx, ny)):
                    crowd_density = sum(grid[mx, my] == PERSON 
                                      for mx, my in get_neighbors(grid, nx, ny))
                    new_dist = distance_to_exit[nx, ny]
                    possible_moves.append((nx, ny, new_dist, crowd_density))
            
            if possible_moves:
                group1 = []
                group2 = []
                group3 = []
                for move in possible_moves:
                    nx, ny, new_dist, crowd_density = move
                    if new_dist < current_dist:
                        group1.append((nx, ny, crowd_density))
                    elif new_dist == current_dist:
                        group2.append((nx, ny, crowd_density))
                    else:
                        group3.append((nx, ny, crowd_density))
                
                if group1:
                    best_group = group1
                elif group2:
                    best_group = group2
                else:
                    best_group = group3
                
                min_density = min(move[2] for move in best_group)
                best_moves = [move for move in best_group if move[2] == min_density]
                nx, ny, _ = best_moves[np.random.randint(len(best_moves))]
                
                if grid[nx, ny] == EXIT:
                    new_grid[x, y] = EMPTY
                elif new_grid[nx, ny] == EMPTY:
                    new_grid[x, y] = EMPTY
                    new_grid[nx, ny] = PERSON
    
    return new_grid

def update(frame):
    """Update function for animation."""
    global grid
    grid = update_fire(grid)
    grid = update_people(grid)
    ax.clear()
    cmap = plt.cm.get_cmap('viridis')
    cmap.set_under('white')
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=4)
    ax.set_title(f"Fire Evacuation Simulation - Frame {frame}")
    return [ax]

# Set up animation
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=100, interval=500, blit=False)
plt.show()