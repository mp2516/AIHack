import os
import numpy as np
import pandas as pd
from tqdm import trange
import shapefile
from matplotlib import pyplot as plt
from kmeans import KMeans2D

baseDir = 'data/california/california/train'

# meta_data = pd.read_csv(os.path.join(baseDir,'BG_METADATA_2016.csv'))
counts_data = pd.read_csv(os.path.join(baseDir, 'X19_INCOME.csv'))
print(counts_data.head())
ca_sf = shapefile.Reader('data/tl_2018_06_tract')
us_sf = shapefile.Reader('data/tl_2018_us_state')
coords = []
inc = []

# Plot california
california_shape = us_sf.shape(13)
x_cali = np.zeros((len(california_shape.points), 1))
y_cali = np.zeros((len(california_shape.points), 1))

for i in range(len(california_shape.points)):
    x_cali[i] = california_shape.points[i][0]
    y_cali[i] = california_shape.points[i][1]

plt.scatter(x_cali, y_cali, s=0.2)

for i in trange(int(len(ca_sf.shapes()))):
    ca_tract_shape = ca_sf.shape(i)
    GEOID = ca_sf.record(i)[3]
    tract_df = counts_data[counts_data['GEOID'].str.contains(GEOID, na=True)]
    if len(tract_df['B19013e1']) == 0:
        continue
    income = sum(tract_df['B19013e1']) / len(tract_df['B19013e1'])
    coords.append((float(ca_sf.record(i)[-1]), float(ca_sf.record(i)[-2])))
    inc.append(income)

km = KMeans2D(coords, 15)
km.iterate(50)
for i in range(15):
    plt.scatter(km.get_x_cluster(i), km.get_y_cluster(i), label=f"Cluster {i}")
plt.legend()
plt.show()

# print(race_data['GEOID'])
