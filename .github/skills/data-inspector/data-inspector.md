---
name: data-inspector
description: Inspects data files, checks for consistency, and outputs summaries of raster properties and feature distributions. Use when you need to understand the input data or verify that rasters are properly aligned and preprocessed. Generates standardized, lightweight Python snippets to visually inspect 2D spatial matrices (NumPy arrays) during simulation development.
---

# Skill: Data Inspector

Whenever the human user or the Tech Lead requests a visual check of the Cellular Automata grid, or when debugging spatial array transformations, you MUST use the following standardized approach to generate a visualization script.

## Core Directives
1. **Use Matplotlib** Always use `matplotlib.pyplot` for generating visualizations.
2. **Handle Masks** If visualizing a risk layer of fire state, explicitly handle the NoData mask (e.g., ocean = 0 or -9999) so it does not skew the color scale. Use `cmap.set_bad(color='black')` for masked areas.
3. **Color Mapping** When plotting the 5-state fire grid, use a discrete colormap that clearly distinguishes the 5 CA states (1: Gray, 2: Green, 3: Yellow, 4: Red, 5: Black).
4. **No side effects** The visualization snippet must NOT alter the underlying matrix data.

## Example Output Snippet
When triggered, generate a standalone block of code similar to this:
```python
import numpy as np
import matplotlib.pyplot as plt

def inspect_grid(grid_array, title="Grid Inspection"):
  plt.figure(figsize=(10,8))
  # Mask out 0s if they represent ocean/nodata
    masked_array = np.ma.masked_where(grid_array == 0, grid_array)
    cmap = plt.cm.viridis
    cmap.set_bad(color='black')
    
    plt.imshow(masked_array, cmap=cmap)
    plt.colorbar(label='Value')
    plt.title(title)
    plt.show()




