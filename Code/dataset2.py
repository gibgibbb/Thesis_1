import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

data = {
    'Building_Presence': np.random.randint(0, 2, n),
    'Building_Material': np.random.choice(['wood', 'concrete'], n),
    'Building_Height': np.random.uniform(0, 50, n),
    'Building_Combustability': np.random.uniform(0.0, 1.0, n),
    'Distance_to_Next_Building': np.random.uniform(0, 100, n),
    'Fuel_Load': np.random.uniform(0.0, 1.0, n),
    'Wind_Speed': np.random.uniform(0, 30, n),
    'Wind_Direction': np.random.uniform(0, 360, n),
    'Temperature': np.random.uniform(15, 45, n),
    'Elevation': np.random.uniform(0, 1000, n),
    'Slope': np.random.uniform(0, 90, n),
    'Aspect': np.random.uniform(0, 360, n),
    'Vegetation_Cover': np.random.uniform(0.0, 1.0, n),
    'Ignited': np.random.randint(0, 2, n),
}

df = pd.DataFrame(data)

# Compute synthetic ignited
# logit = (
#     0.5 * df['Building_Presence'] +
#     2 * df['Vegetation_Cover'] +
#     1.5 * df['Fuel_Load'] +
#     0.1 * df['Wind_Speed'] +
#     0.05 * df['Temperature'] -
#     0.3 * df['Neighborhood_Context'] -
#     0.01 * df['Distance_to_Next_Building'] +
#     (np.where(df['Building_Material'] == 'wood', 1, 0)) * 2
# )

# prob = 1 / (1 + np.exp(-logit))
# df['Ignited'] = np.random.binomial(1, prob)

df = pd.DataFrame(data)
df.to_csv('test.csv', index=False)