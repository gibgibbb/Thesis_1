import pandas as pd
import numpy as np

np.random.seed(42)
n_rows = 1000
data = {
    'Building_Material': np.random.choice(['Wood', 'Concrete', 'Steel', 'Brick'], n_rows),
    'Wind_Speed': np.random.uniform(0, 20, n_rows),
    'Wind_Direction': np.random.uniform(0, 360, n_rows),
    'Slope': np.random.uniform(-20, 20, n_rows),
    'Fuel_Load': np.random.uniform(0, 10, n_rows),
    'Temperature': np.random.uniform(20, 50, n_rows),
    'Humidity': np.random.uniform(10, 100, n_rows),
    'Neighbor_Burning': np.random.randint(0, 9, n_rows),
    'Ignited': np.random.choice([0, 1], n_rows, p=[0.7, 0.3])
}
df = pd.DataFrame(data)
df.to_csv('synthetic_dataset.csv', index=False)