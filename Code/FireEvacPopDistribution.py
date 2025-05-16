import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# Step 2: Define the simulation area (1km x 1km in Lower Manhattan)
# Coordinates in NY State Plane (feet), convert to meters (1 ft = 0.3048 m)
MIN_X = 980000  # Approx min X in Financial District (feet)
MAX_X = 983282  # MIN_X + 1000m (3282 ft)
MIN_Y = 194000  # Approx min Y (feet)
MAX_Y = 197282  # MIN_Y + 1000m (3282 ft)
GRID_SIZE = 100  # 100x100 grid
CELL_SIZE = 10  # 10m per cell

# Step 3: Load the dataset
df = pd.read_csv('./BuildingLayout/MN.csv') 

# Step 4: Filter buildings within the area
df = df[(df['XCoord'] >= MIN_X) & (df['XCoord'] <= MAX_X) &
        (df['YCoord'] >= MIN_Y) & (df['YCoord'] <= MAX_Y)]

# Step 5: Map coordinates to grid cells
def coords_to_grid(x, y, min_x, max_x, min_y, max_y, grid_size):
    # Convert feet to meters and map to grid
    x_meters = x * 0.3048
    y_meters = y * 0.3048
    min_x_m = min_x * 0.3048
    max_x_m = max_x * 0.3048
    min_y_m = min_y * 0.3048
    max_y_m = max_y * 0.3048
    i = int((max_y_m - y_meters) / (max_y_m - min_y_m) * grid_size)
    j = int((x_meters - min_x_m) / (max_x_m - min_x_m) * grid_size)
    return max(0, min(i, grid_size-1)), max(0, min(j, grid_size-1))

building_positions = []
for _, row in df.iterrows():
    if pd.notna(row['XCoord']) and pd.notna(row['YCoord']):
        i, j = coords_to_grid(row['XCoord'], row['YCoord'], MIN_X, MAX_X, MIN_Y, MAX_Y, GRID_SIZE)
        building_positions.append((i, j))

# States
EMPTY = 0
PERSON = 1
FIRE = 2
EXIT = 3
EVACUATED = 4
BUILDING = 5

# Step 6: Initialize grid with buildings
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
for pos in building_positions:
    grid[pos] = BUILDING

# Set exits 
EXIT_POINTS = [(0, 0), (0, 50), (0, 99), (99, 50)]
for exit_pos in EXIT_POINTS:
    if grid[exit_pos] != BUILDING:
        grid[exit_pos] = EXIT

# Set initial fire (example position)
FIRE_START = (50, 50)
if grid[FIRE_START] != BUILDING:
    grid[FIRE_START] = FIRE

# Populate with people avoiding buildings and fire
person_positions = []
while len(person_positions) < 100:
    pos = np.random.randint(0, GRID_SIZE, size=2)
    if grid[tuple(pos)] == EMPTY:
        grid[tuple(pos)] = PERSON
        person_positions.append(tuple(pos))

# Simulation parameters
P_BURN = 0.2 #(.58) ang normal ako gi pa ubos
MOVE_PROB = 0.8

# Precompute distance to nearest exit
distance_to_exit = np.full((GRID_SIZE, GRID_SIZE), np.inf)
for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        if grid[x, y] != BUILDING:
            distances = [abs(x - ex) + abs(y - ey) for ex, ey in EXIT_POINTS]
            distance_to_exit[x, y] = min(distances)

def get_neighbors(grid, x, y, radius=1):
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
    for nx, ny in get_neighbors(grid, x, y):
        if grid[nx, ny] == FIRE:
            return False
    return True

# Step 7: Update fire, preventing spread to buildings
def update_fire(grid):
    new_grid = grid.copy()
    fire_cells = np.where(grid == FIRE)
    for x, y in zip(fire_cells[0], fire_cells[1]):
        for nx, ny in get_neighbors(grid, x, y):
            if grid[nx, ny] in [EMPTY, PERSON] and grid[nx, ny] != BUILDING and np.random.random() < P_BURN:
                new_grid[nx, ny] = FIRE
    return new_grid

def update_people(grid):
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
                    crowd_density = sum(grid[mx, my] == PERSON for mx, my in get_neighbors(grid, nx, ny))
                    new_dist = distance_to_exit[nx, ny]
                    possible_moves.append((nx, ny, new_dist, crowd_density))
            if possible_moves:
                group1 = [m for m in possible_moves if m[2] < current_dist]
                group2 = [m for m in possible_moves if m[2] == current_dist]
                group3 = [m for m in possible_moves if m[2] > current_dist]
                best_group = group1 or group2 or group3
                min_density = min(move[3] for move in best_group)
                best_moves = [move for move in best_group if move[3] == min_density]
                nx, ny, _, _ = best_moves[np.random.randint(len(best_moves))]
                if grid[nx, ny] == EXIT:
                    new_grid[x, y] = EMPTY
                elif new_grid[nx, ny] == EMPTY:
                    new_grid[x, y] = EMPTY
                    new_grid[nx, ny] = PERSON
    return new_grid

# Step 8: Animation with updated visualization
def update(frame):
    global grid
    grid = update_fire(grid)
    grid = update_people(grid)
    ax.clear()
    cmap = plt.cm.get_cmap('tab10')
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=5)
    ax.set_title(f"Fire Evacuation Simulation - Frame {frame}")
    return [ax]

fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=100, interval=500, blit=False)
plt.show()