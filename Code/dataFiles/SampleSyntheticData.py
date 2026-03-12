import numpy as np

# Define the 10x10 grid
grid = np.array([
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 2, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 1, 1, 2, 1, 1, 2, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 2, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 3, 3, 3, 0, 0, 0, 0]
])

# Initial evacuee positions (10 people per building)
evacuees = [(1, 4), (4, 1), (4, 4), (4, 7), (7, 4)]  # Building cell coordinates

# Fire position (static for simplicity)
fire = [(4, 4)]

# Evacuation points
exits = [(9, 3), (9, 4), (9, 5)]

# Print the grid to verify
print("Urban Grid Layout:")
print(grid)
print("\nEvacuee Starting Positions:", evacuees)
print("Fire Location:", fire)