import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# States: 0=empty, 1=tree, 2=fire
neighbourhood = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
p, f = 0.01, 0.0001  # Growth and ignition probabilities

def iterate(grid):
    new_grid = np.zeros_like(grid)
    for i in range(1, grid.shape[0]-1):
        for j in range(1, grid.shape[1]-1):
            if grid[i,j] == 2:
                new_grid[i,j] = 0
            elif grid[i,j] == 0 and np.random.rand() < p:
                new_grid[i,j] = 1
            elif grid[i,j] == 1:
                new_grid[i,j] = 1
                ignited = False
                for di, dj in neighbourhood:
                    if abs(di) == abs(dj) and np.random.rand() < 0.573: continue
                    if grid[i+di, j+dj] == 2:
                        ignited = True
                        break
                if ignited or np.random.rand() < f:
                    new_grid[i,j] = 2
    return new_grid

# Initialize grid
grid = np.zeros((100, 100))
grid[1:-1, 1:-1] = np.random.choice([0,1], size=(98,98), p=[0.8, 0.2])
grid[50,50] = 2  # Ignition point

# Animation setup (similar to examples in sources)
fig, ax = plt.subplots()
im = ax.imshow(grid, cmap='viridis')
def update(frame):
    global grid
    grid = iterate(grid)
    im.set_data(grid)
    return [im]
ani = animation.FuncAnimation(fig, update, frames=200, interval=100)
plt.show()