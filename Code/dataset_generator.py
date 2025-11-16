import numpy as np
import pandas as pd
from scipy.special import expit 
import os

OUT = "revised_dataset.csv"
np.random.seed(42)

# Grid size (adjust)
nx, ny = 50, 40   # 2000 cells
N = nx * ny

# Create smooth spatial fields by cumulative sum smoothing (simple)
base_noise = np.random.randn(ny, nx)
smooth = np.cumsum(np.cumsum(base_noise, axis=0), axis=1)
smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())

# Coordinates
xs, ys = np.meshgrid(np.arange(nx), np.arange(ny))
xs = xs.astype(float); ys = ys.astype(float)

# Building presence probability varies with smooth field
building_prob = 0.2 + 0.5 * smooth  # more buildings where smooth is high
building_presence = (np.random.rand(ny, nx) < building_prob).astype(int)

# Building material (wood more likely where building_density high)
material_choices = ['wood', 'concrete']
material = np.full((ny,nx), "concrete", dtype=object)
for i in range(ny):
    for j in range(nx):
        if building_presence[i,j]==1:
            # Higher probability for concrete (0.7) than wood (0.3 * smooth)
            p_wood = 0.3 * smooth[i,j]
            p_concrete = 0.7
            p = [p_wood, p_concrete]
            p = np.array(p); p = p / p.sum()  # Normalize to sum to 1
            material[i,j] = np.random.choice(material_choices, p=p)
        else:
            material[i,j] = 'none'

# Building height correlated with building presence and smoothness
height = (5 + 40 * smooth) * building_presence + np.random.rand(ny,nx) * 3

# Fuel load correlated with vegetation cover (complement of building presence) and smooth
vegetation = (1 - building_presence) * (0.3 + 0.7 * (1 - smooth))  # more veg where smooth low
fuel_load = np.clip(vegetation + 0.1 * np.random.rand(ny,nx), 0, 1)

# Wind: global constant for simplicity, but direction matters for neighbor alignment
wind_speed = 8.0  # m/s
wind_dir_deg = 45.0  # NE
wind_rad = np.deg2rad(wind_dir_deg)

# Temperature and humidity — add spatial gradient
temperature = 25 + 10 * (1 - smooth) + np.random.randn(ny,nx) * 1.5
humidity = 40 + 30 * smooth + np.random.randn(ny,nx) * 5
humidity = np.clip(humidity, 0, 100)

# Slope: derived from smooth gradients approximated by gradients of smooth map
gy, gx = np.gradient(smooth)
slope = np.sqrt(gx**2 + gy**2) * 30  # scale to degrees approximately

# Material flammability map (only wood and concrete now)
mat_score_map = {'wood':0.95, 'concrete':0.2, 'none':0.05}

# Composite flammability
material_score = np.vectorize(lambda m: mat_score_map.get(m, 0.5))(material)
fuel_moisture = (humidity / 100.0) * (1 - (temperature - temperature.min()) / (temperature.max()-temperature.min()))
fuel_moisture = np.clip(fuel_moisture, 0, 1)
composite_flammability = material_score * fuel_load * (1 - fuel_moisture)

# CA-style simulation to generate labels:
timesteps = 5
state = np.zeros((ny,nx), dtype=int)  # 0=unburned,1=burning,2=burned
# seed a few ignition points proportional to high flammability
seed_prob = (composite_flammability.flatten() - composite_flammability.min())
seed_prob = seed_prob / (seed_prob.max() + 1e-9)
seed_idx = np.random.choice(range(N), size=10, replace=False, p=seed_prob/seed_prob.sum())
for idx in seed_idx:
    i = idx // nx; j = idx % nx
    state[i,j] = 1  # burning

# neighbors offsets (8-neighborhood)
neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# logistic weights (tunable)
w_flamm = 3.0
w_neigh = 1.5
w_wind = 1.0
w_slope = 0.8
bias = -3.0  # keep base P low (rare ignition)

# create arrays to store per-cell record for a random timestep snapshot (we'll store features at t and label whether they ignite at t+1)
records = []

for t in range(timesteps):
    # compute ignition probability for unburned cells (0)
    prob = np.zeros_like(composite_flammability)
    for i in range(ny):
        for j in range(nx):
            if state[i,j] != 0:
                prob[i,j] = 0.0
                continue
            # neighbor influence: count burning neighbors and compute weighted alignment with wind
            nb_burning = 0
            wind_align_sum = 0.0
            for di,dj in neigh:
                ni = i+di; nj = j+dj
                if 0 <= ni < ny and 0 <= nj < nx:
                    if state[ni,nj] == 1:
                        nb_burning += 1
                        # vector from neighbor to this cell
                        vec_x = j - nj
                        vec_y = i - ni
                        # normalize
                        norm = np.hypot(vec_x, vec_y) if (vec_x or vec_y) else 1.0
                        vec_x /= norm; vec_y /= norm
                        # wind unit vector
                        wx = np.cos(wind_rad); wy = np.sin(wind_rad)
                        wind_align_sum += (vec_x*wx + vec_y*wy)  # cosine
            wind_align = wind_align_sum / (nb_burning+1e-6)

            flamm = composite_flammability[i,j]
            slope_comp = slope[i,j]  # rough proxy; you can make directional later

            linear = (w_flamm * flamm) + (w_neigh * nb_burning) + (w_wind * wind_align * (np.linalg.norm([wind_speed,0]))) + (w_slope * slope_comp/30.0) + bias
            p = expit(linear)  # logistic
            prob[i,j] = p

    # apply stochastic ignition: burning cells at next step
    new_burning = (np.random.rand(ny,nx) < prob).astype(int)
    # save records for all cells at this timestep (features at t and whether ignite at t+1)
    for i in range(ny):
        for j in range(nx):
            rec = {
                'x': j,
                'y': i,
                'Building_Presence': int(building_presence[i,j]),
                'Building_Material': material[i,j],
                'Building_Height': float(height[i,j]),
                'Fuel_Load': float(fuel_load[i,j]),
                'Wind_Speed': float(wind_speed),
                'Wind_Direction': float(wind_dir_deg),
                'Temperature': float(temperature[i,j]),
                'Humidity': float(humidity[i,j]),
                'Slope': float(slope[i,j]),
                'Neighbor_Burning': int(sum(1 for di,dj in neigh if 0<=i+di<ny and 0<=j+dj<nx and state[i+di,j+dj]==1)),
                'Ignited': int(new_burning[i,j]),
                'composite_flamm': float(composite_flammability[i,j]),
                'fuel_moisture': float(fuel_moisture[i,j])
            }
            records.append(rec)

    # update states: burning -> burned, new_burning -> burning
    state = np.where(state==1, 2, state)
    state = np.where(new_burning==1, 1, state)

# assemble dataframe and save
df = pd.DataFrame(records)
df.to_csv(OUT, index=False)
print("Generated realistic synthetic dataset saved to:", OUT)